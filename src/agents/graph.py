"""Streamlined LangGraph Multi-Agent Orchestration Engine.

Design Principles:
1. Simple & bounded: Linear StateGraph with strict max iterations (recursion limit).
2. Zero data redundancy: Deduplicates reasons and evidence items in-place.
3. Explicit Human-In-The-Loop routing for missing fields or variances.
4. Idempotent & Side-Effect Free: run_audit does NOT pollute or advance the ledger on read.
   Committing to the ledger only happens on explicit human approval or commit action.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, TypedDict
from langgraph.graph import StateGraph, END

from src.core.schema import (
    AuditResult,
    Decision,
    EvidenceItem,
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    Reason,
    Severity,
)
from src.core.arithmetic import check_missing_fields, validate_arithmetic
from src.core.matcher import CumulativeLedger, DeterministicMatcher
from src.rag.contract_rag import ContractRAGStore

# Bounded iteration ceiling to prevent runaway graph execution
MAX_GRAPH_ITERATIONS = 5


class AuditGraphState(TypedDict):
    """Compact shared state passed between LangGraph nodes without data duplication."""
    invoice: Invoice
    po: Optional[PurchaseOrder]
    receipt: Optional[GoodsReceipt]
    reasons: List[Reason]
    evidence_bundle: List[EvidenceItem]
    financial_exposure: Decimal
    risk_score: float
    decision: Decision
    requires_human_review: bool
    human_in_loop_message: Optional[str]
    investigation_summary: str
    matched_lines_map: Dict[int, int]


def deduplicate_reasons(reasons: List[Reason]) -> List[Reason]:
    """Remove redundant reasons based on (code, message) uniqueness."""
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Reason] = []
    for r in reasons:
        key = (r.code.value, r.message)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def deduplicate_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """Remove redundant citations based on (source_document, reference_id)."""
    seen: Set[Tuple[str, str]] = set()
    deduped: List[EvidenceItem] = []
    for e in evidence:
        key = (e.source_document, e.reference_id)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


class AuditWorkflow:
    """Builds and executes the bounded LangGraph invoice auditor."""

    def __init__(
        self,
        ledger: Optional[CumulativeLedger] = None,
        contract_rag: Optional[ContractRAGStore] = None,
    ) -> None:
        self.ledger = ledger or CumulativeLedger()
        self.contract_rag = contract_rag or ContractRAGStore()
        self.matcher = DeterministicMatcher(self.ledger)
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AuditGraphState)

        # 3 Simple, focused nodes
        graph.add_node("validator", self._validator_node)
        graph.add_node("reconciler", self._reconciler_node)
        graph.add_node("contract_and_triage", self._contract_and_triage_node)

        # Direct linear flow
        graph.set_entry_point("validator")
        graph.add_edge("validator", "reconciler")
        graph.add_edge("reconciler", "contract_and_triage")
        graph.add_edge("contract_and_triage", END)

        return graph.compile()

    def _validator_node(self, state: AuditGraphState) -> Dict:
        """Node 1: Validates required fields (HITL on missing) and checks math."""
        invoice = state["invoice"]
        reasons = list(state.get("reasons", []))
        financial_exposure = state.get("financial_exposure", Decimal("0.00"))

        # Missing required fields trigger Human-In-The-Loop review
        reasons.extend(check_missing_fields(invoice))

        # Deterministic arithmetic integrity check
        arith_reasons, arith_variance = validate_arithmetic(invoice)
        reasons.extend(arith_reasons)
        financial_exposure += arith_variance

        evidence = list(state.get("evidence_bundle", []))
        if arith_reasons:
            evidence.append(
                EvidenceItem(
                    category="Arithmetic Integrity",
                    source_document="Invoice Document",
                    reference_id=invoice.invoice_number or "UNASSIGNED",
                    citation_text=f"Arithmetic variance of ${arith_variance} detected across lines/totals.",
                    confidence=1.0,
                )
            )

        return {
            "reasons": deduplicate_reasons(reasons),
            "financial_exposure": financial_exposure,
            "evidence_bundle": deduplicate_evidence(evidence),
        }

    def _reconciler_node(self, state: AuditGraphState) -> Dict:
        """Node 2: Deterministic 2-Way & 3-Way match against PO and Goods Receipt."""
        invoice = state["invoice"]
        po = state["po"]
        receipt = state["receipt"]
        reasons = list(state["reasons"])
        financial_exposure = state["financial_exposure"]
        evidence = list(state["evidence_bundle"])

        match_reasons, match_variance, matched_lines = self.matcher.match(invoice, po, receipt)
        reasons.extend(match_reasons)
        financial_exposure += match_variance

        if po:
            evidence.append(
                EvidenceItem(
                    category="PO Reconciliation",
                    source_document=f"Purchase Order #{po.po_number}",
                    reference_id=po.po_number,
                    citation_text=f"PO issued to '{po.vendor}' with {len(po.lines)} lines.",
                    confidence=1.0,
                )
            )

        if receipt:
            evidence.append(
                EvidenceItem(
                    category="Goods Receipt",
                    source_document=f"Receipt #{receipt.receipt_id}",
                    reference_id=receipt.receipt_id,
                    citation_text=f"Verified receipt on {receipt.receipt_date}.",
                    confidence=1.0,
                )
            )

        return {
            "reasons": deduplicate_reasons(reasons),
            "financial_exposure": financial_exposure,
            "evidence_bundle": deduplicate_evidence(evidence),
            "matched_lines_map": matched_lines,
        }

    def _contract_and_triage_node(self, state: AuditGraphState) -> Dict:
        """Node 3: Contract RAG analysis, discrepancy synthesis, and triage rating."""
        invoice = state["invoice"]
        reasons = list(state["reasons"])
        evidence = list(state["evidence_bundle"])
        financial_exposure = state["financial_exposure"]

        # Contract RAG clause audit
        contract_reasons, contract_evidence, violation_amt = self.contract_rag.audit_contract_compliance(invoice)
        reasons.extend(contract_reasons)
        evidence.extend(contract_evidence)
        financial_exposure += violation_amt

        # Deduplicate to minimize data redundancy
        clean_reasons = deduplicate_reasons(reasons)
        clean_evidence = deduplicate_evidence(evidence)

        has_reject = any(r.severity == Severity.REJECT for r in clean_reasons)
        has_review = any(r.severity == Severity.REVIEW for r in clean_reasons)
        hitl_reasons = [r for r in clean_reasons if r.requires_hitl]

        if has_reject:
            decision = Decision.REJECT
            risk_score = min(100.0, 80.0 + float(len(clean_reasons) * 4))
            requires_hitl = True
        elif has_review or hitl_reasons:
            decision = Decision.NEEDS_REVIEW
            risk_score = min(79.0, 35.0 + float(len(clean_reasons) * 8))
            requires_hitl = True
        else:
            decision = Decision.AUTO_APPROVE
            risk_score = 5.0
            requires_hitl = False

        # Compact summary
        if decision == Decision.AUTO_APPROVE:
            summary = "Audit Passed: All 3-way reconciliation, arithmetic checks, and contract terms verified."
        else:
            summary = f"Audit {decision.value.upper()}: {len(clean_reasons)} discrepancy notice(s). Total exposure: ${financial_exposure}."

        # Actionable Human-In-The-Loop guidance
        hitl_msg = None
        if requires_hitl:
            hitl_items = [r.message for r in hitl_reasons if "Missing" in r.message or "Human-In-The-Loop" in r.message]
            if hitl_items:
                hitl_msg = hitl_items[0]
            else:
                hitl_msg = f"Review required. Discrepancy financial exposure: ${financial_exposure}. Human verification required."

        return {
            "reasons": clean_reasons,
            "evidence_bundle": clean_evidence,
            "financial_exposure": financial_exposure,
            "decision": decision,
            "risk_score": round(risk_score, 1),
            "requires_human_review": requires_hitl,
            "human_in_loop_message": hitl_msg,
            "investigation_summary": summary,
        }

    def run_audit(
        self,
        invoice: Invoice,
        po: Optional[PurchaseOrder] = None,
        receipt: Optional[GoodsReceipt] = None,
    ) -> AuditResult:
        """Run the bounded LangGraph audit pipeline with strict max iterations.
        
        Note: This method is purely IDEMPOTENT. It audits against current ledger state
        without mutating or advancing billing counts, preventing self-duplicate traps.
        """
        initial_state: AuditGraphState = {
            "invoice": invoice,
            "po": po,
            "receipt": receipt,
            "reasons": [],
            "evidence_bundle": [],
            "financial_exposure": Decimal("0.00"),
            "risk_score": 0.0,
            "decision": Decision.NEEDS_REVIEW,
            "requires_human_review": False,
            "human_in_loop_message": None,
            "investigation_summary": "",
            "matched_lines_map": {},
        }

        # Max iterations ceiling
        final_state = self.app.invoke(
            initial_state,
            config={"recursion_limit": MAX_GRAPH_ITERATIONS},
        )

        return AuditResult(
            invoice_id=invoice.invoice_number or "UNASSIGNED",
            decision=final_state["decision"],
            reasons=final_state["reasons"],
            evidence_bundle=final_state["evidence_bundle"],
            risk_score=final_state["risk_score"],
            requires_human_review=final_state["requires_human_review"],
            human_in_loop_message=final_state["human_in_loop_message"],
            financial_exposure=final_state["financial_exposure"],
            investigation_summary=final_state["investigation_summary"],
            matched_lines_map=final_state.get("matched_lines_map", {}),
        )

    def commit_invoice_to_ledger(
        self,
        invoice: Invoice,
        po: Optional[PurchaseOrder] = None,
        matched_lines_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """Explicitly commit an approved invoice to the cumulative ledger.
        
        Properly maps invoice lines to authorized PO lines to prevent index corruption.
        """
        self.ledger.record_invoice(invoice)
        if po:
            lines_map = matched_lines_map or {}
            for line in invoice.lines:
                # Map to matched PO line number, fallback to line.line_number
                po_line_num = lines_map.get(line.line_number, line.line_number)
                self.ledger.record_po_billing(po.po_number, po_line_num, line.quantity)
