"""Session Manager for Isolated, Persistent Multi-Agent Procurement Audits.

Provides:
- Unique session ID and folder persistence: `data/sessions/{session_id}/`
- Step-by-step parallelized audit execution:
    1. Parallel Contract Chunking & Vector DB Indexing
    2. Parallel Dynamic PDF Ingestion & Coordinate/LLM Extraction
    3. Multi-Agent 3-Way Reconciliation & Contract Vector RAG
    4. Stateful Cumulative Ledger & Composite History Verification
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.core.schema import (
    AuditResult,
    ContractClause,
    Decision,
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    ReviewDecision,
)
from src.core.matcher import CumulativeLedger
from src.rag.contract_rag import ContractRAGStore
from src.agents.graph import AuditWorkflow
from src.extractor.pdf_extractor import PDFExtractor
from src.synth.dataset_gen import generate_synthetic_dataset
from src.core.table_parser import parse_contract_clauses, parse_goods_receipts, parse_purchase_orders


class AuditSession:
    """An isolated, fully persistent procurement audit session."""

    def __init__(self, session_id: str, base_dir: Path) -> None:
        self.session_id = session_id
        self.session_dir = base_dir / session_id
        self.invoices_dir = self.session_dir / "invoices"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "created"  # created -> chunking_rag -> extracting_invoices -> auditing -> completed | error

        # Session-specific isolated singletons & stores
        self.contract_rag = ContractRAGStore()
        self.ledger = CumulativeLedger()
        self.workflow = AuditWorkflow(ledger=self.ledger, contract_rag=self.contract_rag)
        self.extractor = PDFExtractor()

        # In-memory document stores
        self.invoices: Dict[str, Invoice] = {}
        self.purchase_orders: Dict[str, PurchaseOrder] = {}
        self.goods_receipts: Dict[str, GoodsReceipt] = {}
        self.audit_results: Dict[str, AuditResult] = {}
        self.human_decisions: Dict[str, ReviewDecision] = {}

        # Workflow step execution tracking
        self.steps: List[Dict[str, Any]] = [
            {"step": 1, "title": "Contract Chunking & Vector RAG Indexing", "status": "pending", "details": ""},
            {"step": 2, "title": "Parallel PDF Extraction & Dynamic Parsing", "status": "pending", "details": ""},
            {"step": 3, "title": "Multi-Agent 3-Way Reconciliation & RAG Audit", "status": "pending", "details": ""},
            {"step": 4, "title": "Cumulative Ledger & Composite History Verification", "status": "pending", "details": ""},
        ]

        # Ensure directory structure
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.invoices_dir.mkdir(parents=True, exist_ok=True)

    def persist_metadata(self) -> None:
        """Persist session state and steps tracking to disk."""
        meta = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status,
            "invoice_count": len(self.invoices),
            "po_count": len(self.purchase_orders),
            "receipt_count": len(self.goods_receipts),
            "clause_count": len(self.contract_rag.clauses),
            "steps": self.steps,
        }
        with open(self.session_dir / "session_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def persist_audit_results(self) -> None:
        """Persist full audit results and ledger to disk."""
        results_dump = {
            inv_id: res.model_dump(mode="json")
            for inv_id, res in self.audit_results.items()
        }
        with open(self.session_dir / "audit_results.json", "w", encoding="utf-8") as f:
            json.dump(results_dump, f, indent=2)

    # --- Step 1: Parallel Contract Chunking & Vector RAG Indexing ---
    async def step_chunk_contracts_parallel(self, contracts_data: List[Dict[str, Any]]) -> None:
        """Chunk contract clauses in parallel and index into session Vector DB."""
        self.steps[0]["status"] = "in_progress"
        self.steps[0]["details"] = f"Chunking {len(contracts_data)} contract clauses into Vector RAG..."

        def index_single_clause(c_dict: Dict[str, Any]) -> Dict[str, Any]:
            clause = ContractClause(**c_dict)
            return self.contract_rag.add_clause(clause)

        # Run chunking & tokenization across threadpool concurrently
        tasks = [asyncio.to_thread(index_single_clause, c) for c in contracts_data]
        if tasks:
            indexed_details = await asyncio.gather(*tasks)
            total_tokens = sum(d["token_count"] for d in indexed_details)
            self.steps[0]["status"] = "completed"
            self.steps[0]["details"] = f"Successfully chunked & indexed {len(indexed_details)} clauses ({total_tokens} tokens) into Vector RAG."
        else:
            self.steps[0]["status"] = "completed"
            self.steps[0]["details"] = "No custom contract clauses provided. Proceeding with standard terms."

        # Persist session contracts
        with open(self.session_dir / "contracts.json", "w", encoding="utf-8") as f:
            json.dump(contracts_data, f, indent=2)
        self.persist_metadata()

    # --- Step 2: Parallel Dynamic PDF Ingestion & Extraction ---
    async def step_extract_invoices_parallel(self, invoice_files: List[Tuple[str, bytes]]) -> None:
        """Extract all uploaded PDF invoices in parallel using concurrent workers."""
        self.steps[1]["status"] = "in_progress"
        self.steps[1]["details"] = f"Extracting {len(invoice_files)} PDF invoices concurrently..."

        def extract_single(fname: str, raw_bytes: bytes) -> Tuple[str, Invoice, List[Any]]:
            # Save raw PDF into session folder
            pdf_path = self.invoices_dir / fname
            with open(pdf_path, "wb") as f:
                f.write(raw_bytes)
            inv, reasons = self.extractor.extract_from_bytes(raw_bytes)
            inv_id = inv.invoice_number or fname
            return inv_id, inv, reasons

        # Execute extraction across all PDFs concurrently in parallel
        tasks = [asyncio.to_thread(extract_single, fname, b) for fname, b in invoice_files]
        results = await asyncio.gather(*tasks)

        for inv_id, inv, _ in results:
            self.invoices[inv_id] = inv

        self.steps[1]["status"] = "completed"
        self.steps[1]["details"] = f"Extracted {len(self.invoices)} invoice documents in parallel with visual horizontal reconstruction."
        self.persist_metadata()

    # --- Step 3 & 4: Multi-Agent Audit, Stateful Ledger & Review Routing ---
    async def step_audit_and_ledger_process(self) -> None:
        """Execute deterministic multi-agent audit and stateful composite history checking."""
        self.steps[2]["status"] = "in_progress"
        self.steps[2]["details"] = f"Auditing {len(self.invoices)} invoices against POs, receipts & contract clauses..."

        def audit_single(inv_id: str, inv: Invoice) -> Tuple[str, AuditResult]:
            po = self.purchase_orders.get(inv.po_number or "")
            receipt = self.goods_receipts.get(inv.po_number or "")
            res = self.workflow.run_audit(invoice=inv, po=po, receipt=receipt)
            return inv_id, res

        # Run multi-agent audit concurrently for all invoices
        tasks = [asyncio.to_thread(audit_single, inv_id, inv) for inv_id, inv in self.invoices.items()]
        audit_outputs = await asyncio.gather(*tasks)

        for inv_id, res in audit_outputs:
            self.audit_results[inv_id] = res

        self.steps[2]["status"] = "completed"
        self.steps[2]["details"] = f"Multi-Agent 3-way matching and contract vector RAG completed for {len(audit_outputs)} invoices."

        # Step 4: Ledger & Composite History Check
        self.steps[3]["status"] = "in_progress"
        self.steps[3]["details"] = "Verifying unique composite history (Address + Name + ID + Date) and recording risk scores..."

        auto_approved = sum(1 for r in self.audit_results.values() if r.decision == Decision.AUTO_APPROVE)
        review_count = sum(1 for r in self.audit_results.values() if r.decision == Decision.NEEDS_REVIEW)
        reject_count = sum(1 for r in self.audit_results.values() if r.decision == Decision.REJECT)

        self.steps[3]["status"] = "completed"
        self.steps[3]["details"] = (
            f"Ledger verification complete: {auto_approved} Auto-Approved, "
            f"{review_count} Queued for HITL Review, {reject_count} Flagged for Rejection."
        )
        self.status = "completed"

        self.persist_audit_results()
        self.persist_metadata()

    def record_human_decision(
        self,
        invoice_id: str,
        reviewer_id: str,
        action: str,
        notes: str,
        overridden_reasons: List[str] = None,
    ) -> ReviewDecision:
        """Record Human-in-the-Loop decision and commit to ledger if approved."""
        decision = ReviewDecision(
            reviewer_id=reviewer_id,
            action=action,
            notes=notes,
            overridden_reasons=overridden_reasons or [],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.human_decisions[invoice_id] = decision

        if invoice_id in self.audit_results:
            self.audit_results[invoice_id].human_decision = decision
            if action == "approve":
                # Commit to session cumulative ledger
                inv = self.invoices.get(invoice_id)
                if inv:
                    po = self.purchase_orders.get(inv.po_number or "")
                    lines_map = self.audit_results[invoice_id].matched_lines_map
                    self.workflow.commit_invoice_to_ledger(inv, po, lines_map)

        self.persist_audit_results()
        return decision


class SessionManager:
    """Manages active and persisted audit sessions."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.sessions_dir = data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, AuditSession] = {}
        self.current_session_id: Optional[str] = None

    def create_session(self, session_id: Optional[str] = None) -> AuditSession:
        """Initialize a new isolated audit session with dedicated disk storage."""
        if not session_id:
            session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        session = AuditSession(session_id=session_id, base_dir=self.sessions_dir)
        self.active_sessions[session_id] = session
        self.current_session_id = session_id
        session.persist_metadata()
        return session

    def get_session(self, session_id: Optional[str] = None) -> Optional[AuditSession]:
        """Retrieve an active or persisted session."""
        target_id = session_id or self.current_session_id
        if not target_id:
            return None

        if target_id in self.active_sessions:
            return self.active_sessions[target_id]

        # Try to restore from disk if exists
        sess_path = self.sessions_dir / target_id
        if sess_path.exists() and (sess_path / "session_metadata.json").exists():
            session = AuditSession(session_id=target_id, base_dir=self.sessions_dir)
            self._load_session_from_disk(session)
            self.active_sessions[target_id] = session
            return session

        return None

    def _load_session_from_disk(self, session: AuditSession) -> None:
        """Restore session stores from disk persistence."""
        meta_file = session.session_dir / "session_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                session.status = meta.get("status", "completed")
                session.steps = meta.get("steps", session.steps)

        # Restore POs from CSV table or JSON
        po_csv = session.session_dir / "purchase_orders.csv"
        po_json = session.session_dir / "purchase_orders.json"
        if po_csv.exists():
            session.purchase_orders = parse_purchase_orders(po_csv.read_bytes())
        elif po_json.exists():
            session.purchase_orders = parse_purchase_orders(po_json.read_bytes())

        # Restore Goods Receipts from CSV table or JSON
        gr_csv = session.session_dir / "goods_receipts.csv"
        gr_json = session.session_dir / "goods_receipts.json"
        if gr_csv.exists():
            session.goods_receipts = parse_goods_receipts(gr_csv.read_bytes())
        elif gr_json.exists():
            session.goods_receipts = parse_goods_receipts(gr_json.read_bytes())

        # Restore Contracts into session Vector RAG dynamically
        c_file = session.session_dir / "contracts.json"
        if c_file.exists():
            for clause in parse_contract_clauses(c_file.read_bytes()):
                session.contract_rag.add_clause(clause)

        # Restore Audit Results
        res_file = session.session_dir / "audit_results.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                raw_res = json.load(f)
                for inv_id, d in raw_res.items():
                    session.audit_results[inv_id] = AuditResult(**d)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions on disk."""
        sessions = []
        for p in sorted(self.sessions_dir.iterdir(), reverse=True):
            if p.is_dir() and (p / "session_metadata.json").exists():
                try:
                    with open(p / "session_metadata.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        sessions.append(meta)
                except Exception:
                    continue
        return sessions
