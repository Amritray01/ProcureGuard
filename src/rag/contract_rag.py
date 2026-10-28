"""Contract & Clause Vector RAG System.

Stores procurement agreements, Master Service Agreements (MSAs), and rate cards.
Uses lowercase semantic tokenization and BM25/cosine retrieval.
Integrates LLM reasoning to interpret contract nuances, terms, and hidden fee policies.
"""

from collections import Counter
from decimal import Decimal
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from src.core.schema import ContractClause, EvidenceItem, Invoice, Reason, ReasonCode, Severity
from src.core.llm import llm_client


def tokenize_lower(text: str) -> List[str]:
    """Tokenize and normalize text to lowercase for robust matching."""
    return re.findall(r"\w+", text.lower())


def compute_bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_dl: float,
    doc_len: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 relevance score between lowercased query and document tokens."""
    doc_counter = Counter(doc_tokens)
    score = 0.0
    for q in query_tokens:
        tf = doc_counter[q]
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / (avg_dl or 1.0)))
            score += (numerator / denominator)
    return score


class ContractRAGStore:
    """Dynamic Vector and Semantic Store for procurement contracts and clauses."""

    def __init__(self) -> None:
        self.clauses: Dict[str, ContractClause] = {}
        self.clause_tokens: Dict[str, List[str]] = {}

    def add_clause(self, clause: ContractClause) -> Dict[str, Any]:
        """Add or update a contract clause and return indexed chunk details."""
        self.clauses[clause.clause_id] = clause
        combined_text = f"{clause.title} {clause.content}".lower()
        tokens = tokenize_lower(combined_text)
        self.clause_tokens[clause.clause_id] = tokens

        return {
            "clause_id": clause.clause_id,
            "vendor": clause.vendor,
            "title": clause.title,
            "token_count": len(tokens),
            "content_preview": clause.content[:160] + "..." if len(clause.content) > 160 else clause.content,
            "prohibits_surcharges": clause.prohibits_surcharges,
            "discount_percentage": str(clause.discount_percentage) if clause.discount_percentage else None,
        }

    def add_clauses(self, clauses: List[ContractClause]) -> List[Dict[str, Any]]:
        """Batch add clauses and return summary details for UI inspection."""
        return [self.add_clause(c) for c in clauses]

    def list_all_clauses(self) -> List[Dict[str, Any]]:
        """List all indexed contract clauses with metadata."""
        return [
            {
                "clause_id": c.clause_id,
                "vendor": c.vendor,
                "title": c.title,
                "content": c.content,
                "token_count": len(self.clause_tokens.get(c.clause_id, [])),
                "prohibits_surcharges": c.prohibits_surcharges,
                "discount_percentage": str(c.discount_percentage) if c.discount_percentage else None,
            }
            for c in self.clauses.values()
        ]

    def retrieve(
        self,
        query: str,
        vendor: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[ContractClause, float]]:
        """Retrieve most relevant clauses converting query and documents to lowercase."""
        if not self.clauses:
            return []

        query_tokens = tokenize_lower(query)
        if not query_tokens:
            return []

        target_ids = []
        for c_id, clause in self.clauses.items():
            if vendor:
                norm_vendor = vendor.strip().lower()
                norm_clause_v = clause.vendor.strip().lower()
                if norm_vendor not in norm_clause_v and norm_clause_v not in norm_vendor:
                    continue
            target_ids.append(c_id)

        if not target_ids:
            target_ids = list(self.clauses.keys())

        avg_dl = sum(len(self.clause_tokens[c_id]) for c_id in target_ids) / len(target_ids)
        scored_results: List[Tuple[ContractClause, float]] = []

        for c_id in target_ids:
            doc_toks = self.clause_tokens[c_id]
            score = compute_bm25_score(query_tokens, doc_toks, avg_dl, len(doc_toks))
            if score > 0.05:
                scored_results.append((self.clauses[c_id], score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def audit_contract_compliance(
        self,
        invoice: Invoice,
    ) -> Tuple[List[Reason], List[EvidenceItem], Decimal]:
        """Audit invoice against contract clauses using LLM understanding."""
        reasons: List[Reason] = []
        evidence: List[EvidenceItem] = []
        violation_amount = Decimal("0.00")

        if not invoice.vendor:
            return reasons, evidence, violation_amount

        # Check for surcharges or extra fees
        if invoice.freight_surcharges and invoice.freight_surcharges > Decimal("0.00"):
            surcharge_matches = self.retrieve(
                query="freight shipping handling fuel surcharge prohibited extra fees delivery",
                vendor=invoice.vendor,
                top_k=2,
            )

            for clause, score in surcharge_matches:
                inv_summary = (
                    f"Vendor: {invoice.vendor}, Invoice Total: {invoice.total}, "
                    f"Freight/Surcharges Billed: {invoice.freight_surcharges}"
                )

                # Call LLM to interpret the clause dynamically
                llm_analysis = llm_client.analyze_contract_clause(
                    invoice_summary=inv_summary,
                    clause_text=clause.content,
                    surcharge_amount=invoice.freight_surcharges,
                )

                if llm_analysis.get("is_violation", False) or clause.prohibits_surcharges:
                    sev = Severity.REJECT if llm_analysis.get("severity") == "reject" else Severity.REVIEW
                    reasons.append(
                        Reason(
                            code=ReasonCode.UNAUTHORIZED_SURCHARGE,
                            severity=sev,
                            message=(
                                f"Unauthorized Surcharge: Invoice charges ${invoice.freight_surcharges} "
                                f"in freight/surcharges. Contract Clause '{clause.clause_id}' ('{clause.title}') "
                                f"analysis: {llm_analysis.get('explanation', 'Clause forbids separate freight.')}"
                            ),
                            details={
                                "clause_id": clause.clause_id,
                                "surcharge_amount": str(invoice.freight_surcharges),
                                "explanation": llm_analysis.get("explanation"),
                            },
                            requires_hitl=True,
                        )
                    )
                    violation_amount += invoice.freight_surcharges

                    evidence.append(
                        EvidenceItem(
                            category="Contract Violation",
                            source_document=f"Master Agreement ({clause.vendor})",
                            reference_id=clause.clause_id,
                            citation_text=f"Clause {clause.clause_id} ({clause.title}): '{clause.content}'",
                            confidence=0.98,
                        )
                    )
                    break

        # Check for agreed volume / early payment discounts
        discount_matches = self.retrieve(
            query="early payment volume discount pricing terms rebate tier",
            vendor=invoice.vendor,
            top_k=1,
        )

        for clause, score in discount_matches:
            if clause.discount_percentage and clause.discount_percentage > Decimal("0.0"):
                if not invoice.discount or invoice.discount == Decimal("0.00"):
                    if invoice.subtotal and invoice.subtotal > Decimal("0.00"):
                        expected_discount = (invoice.subtotal * (clause.discount_percentage / Decimal("100.0"))).quantize(Decimal("0.01"))
                        reasons.append(
                            Reason(
                                code=ReasonCode.DISCOUNT_NOT_APPLIED,
                                severity=Severity.REVIEW,
                                message=(
                                    f"Contract Clause '{clause.clause_id}' stipulates a "
                                    f"{clause.discount_percentage}% discount (approx ${expected_discount}), "
                                    f"which is missing from this invoice. Human-In-The-Loop review requested."
                                ),
                                details={
                                    "clause_id": clause.clause_id,
                                    "agreed_discount_pct": str(clause.discount_percentage),
                                    "expected_savings": str(expected_discount),
                                },
                                requires_hitl=True,
                            )
                        )
                        violation_amount += expected_discount

                        evidence.append(
                            EvidenceItem(
                                category="Discount Compliance",
                                source_document=f"Rate Schedule ({clause.vendor})",
                                reference_id=clause.clause_id,
                                citation_text=f"Clause {clause.clause_id}: '{clause.content}'",
                                confidence=0.95,
                            )
                        )

        return reasons, evidence, violation_amount
