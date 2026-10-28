"""FastAPI Web Service for Procurement Invoice & Contract Auditor.

Exposes REST endpoints for:
- Immediate PDF document upload & LangGraph multi-agent auditing
- ERP Tabular Data: Purchase Orders & Goods Receipts tables
- Vector DB: Contract clauses addition and live details inspection
- Human-in-the-Loop review actions (Approve, Flag for correction, Reject)
- LLM-powered Auditor Chatbot copilot
- Synthetic dataset generation
"""

from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.schema import (
    AuditResult,
    ContractClause,
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    ReviewAction,
    ReviewDecision,
)
from src.extractor.pdf_extractor import PDFExtractor
from src.core.matcher import CumulativeLedger
from src.rag.contract_rag import ContractRAGStore
from src.agents.graph import AuditWorkflow
from src.hitl.review_manager import HITLReviewManager
from src.synth.dataset_gen import generate_synthetic_dataset
from src.core.llm import llm_client

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Procurement Invoice & Contract Auditor API",
    description="Deterministic 3-Way Math + Contract Vector RAG + LangGraph Multi-Agents",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
ledger = CumulativeLedger()
contract_rag = ContractRAGStore()
workflow = AuditWorkflow(ledger=ledger, contract_rag=contract_rag)
hitl_manager = HITLReviewManager(log_path=DATA_DIR / "audit_log.json")
extractor = PDFExtractor()

# ERP Stores (Table formats for POs and Receipts)
po_store: Dict[str, PurchaseOrder] = {}
gr_store: Dict[str, GoodsReceipt] = {}
audit_cache: Dict[str, AuditResult] = {}


def load_fixtures():
    """Load default dataset fixtures into memory."""
    po_file = DATA_DIR / "purchase_orders.json"
    gr_file = DATA_DIR / "goods_receipts.json"
    contracts_file = DATA_DIR / "contracts.json"

    if not po_file.exists():
        generate_synthetic_dataset(DATA_DIR)

    if po_file.exists():
        with open(po_file, "r", encoding="utf-8") as f:
            for p in json.load(f):
                po = PurchaseOrder(**p)
                po_store[po.po_number] = po

    if gr_file.exists():
        with open(gr_file, "r", encoding="utf-8") as f:
            for g in json.load(f):
                gr = GoodsReceipt(**g)
                gr_store[gr.po_number] = gr

    if contracts_file.exists():
        with open(contracts_file, "r", encoding="utf-8") as f:
            for c in json.load(f):
                contract_rag.add_clause(ContractClause(**c))


load_fixtures()


class ReviewRequest(BaseModel):
    invoice_id: str
    reviewer_id: str = "Auditor_Admin"
    action: ReviewAction
    notes: str
    overridden_reasons: List[str] = []


class ChatRequest(BaseModel):
    message: str
    invoice_id: Optional[str] = None


class AddClauseRequest(BaseModel):
    clause_id: str
    vendor: str
    title: str
    content: str
    category: str = "pricing"
    prohibits_surcharges: bool = False
    discount_percentage: Optional[Decimal] = None


# --- ERP Tabular Endpoints (PO & Goods Receipts) ---

@app.get("/api/erp/purchase-orders")
def get_purchase_orders_table():
    """Return Purchase Orders in tabular enterprise ERP format."""
    rows = []
    for po in po_store.values():
        for line in po.lines:
            rows.append({
                "po_number": po.po_number,
                "vendor": po.vendor,
                "currency": po.currency,
                "receipt_required": po.receipt_required,
                "line_number": line.line_number,
                "sku": line.sku or "N/A",
                "description": line.description,
                "quantity": str(line.quantity),
                "unit_price": str(line.unit_price),
                "line_total": str((line.quantity * line.unit_price).quantize(Decimal("0.01"))),
            })
    return rows


@app.get("/api/erp/goods-receipts")
def get_goods_receipts_table():
    """Return Goods Receipts in tabular enterprise warehouse format."""
    rows = []
    for gr in gr_store.values():
        for line in gr.lines:
            rows.append({
                "receipt_id": gr.receipt_id,
                "po_number": gr.po_number,
                "receipt_date": gr.receipt_date,
                "line_number": line.line_number,
                "sku": line.sku or "N/A",
                "quantity_received": str(line.quantity_received),
                "status": "Verified in Warehouse",
            })
    return rows


# --- Vector DB Endpoints ---

@app.get("/api/rag/clauses")
def list_vector_db_clauses():
    """List all indexed contract clauses in the Vector DB with details."""
    return contract_rag.list_all_clauses()


@app.post("/api/rag/clauses")
def add_vector_db_clause(req: AddClauseRequest):
    """Add a custom contract clause to the Vector DB and return details immediately."""
    clause = ContractClause(
        clause_id=req.clause_id,
        vendor=req.vendor,
        title=req.title,
        content=req.content,
        category=req.category,
        prohibits_surcharges=req.prohibits_surcharges,
        discount_percentage=req.discount_percentage,
    )
    details = contract_rag.add_clause(clause)
    return {"status": "indexed", "clause_details": details}


# --- Ingestion & LangGraph Audit Endpoints ---

@app.post("/api/audit/pdf")
async def audit_uploaded_pdf(file: UploadFile = File(...)):
    """Upload custom invoice PDF and immediately execute LangGraph multi-agent audit."""
    contents = await file.read()
    invoice, extract_reasons = extractor.extract_from_bytes(contents)

    po = po_store.get(invoice.po_number or "")
    receipt = gr_store.get(invoice.po_number or "")

    inv_id = invoice.invoice_number or file.filename or "CUSTOM_INVOICE"
    result = workflow.run_audit(invoice=invoice, po=po, receipt=receipt)
    audit_cache[inv_id] = result

    return result.model_dump(mode="json")


@app.post("/api/audit/sample/{invoice_id}")
def audit_sample_endpoint(invoice_id: str):
    """Run full LangGraph audit on a sample benchmark invoice."""
    manifest_file = DATA_DIR / "manifest.json"
    if not manifest_file.exists():
        generate_synthetic_dataset(DATA_DIR)
        load_fixtures()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_inv = next((inv for inv in manifest["invoices"] if inv["id"] == invoice_id), None)
    if not target_inv:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found.")

    pdf_path = DATA_DIR / "invoices" / target_inv["file"]
    invoice, extract_reasons = extractor.extract_from_path(pdf_path)

    po = po_store.get(invoice.po_number or "")
    receipt = gr_store.get(invoice.po_number or "")

    result = workflow.run_audit(invoice=invoice, po=po, receipt=receipt)
    audit_cache[invoice_id] = result

    human_dec = hitl_manager.get_decision(invoice_id)
    if human_dec:
        result.human_decision = ReviewDecision(**human_dec)

    return result.model_dump(mode="json")


@app.post("/api/dataset/generate")
def generate_dataset_endpoint():
    """Generate synthetic invoices, POs, Receipts, and Contracts."""
    manifest = generate_synthetic_dataset(DATA_DIR)
    load_fixtures()
    return {"status": "success", "manifest": manifest}


@app.get("/api/dataset/samples")
def get_sample_invoices():
    """List available sample invoices with expected labels."""
    manifest_file = DATA_DIR / "manifest.json"
    if not manifest_file.exists():
        generate_synthetic_dataset(DATA_DIR)
        load_fixtures()

    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Human-in-the-Loop & Chatbot Endpoints ---

@app.post("/api/review")
def record_human_decision(req: ReviewRequest):
    """Record Human-in-the-Loop decision."""
    decision = hitl_manager.record_decision(
        invoice_id=req.invoice_id,
        reviewer_id=req.reviewer_id,
        action=req.action,
        notes=req.notes,
        overridden_reasons=req.overridden_reasons,
    )

    if req.invoice_id in audit_cache:
        audit_cache[req.invoice_id].human_decision = decision

    return {"status": "recorded", "decision": decision.model_dump()}


@app.post("/api/chat")
def auditor_chatbot(req: ChatRequest):
    """LLM-backed conversational copilot with Vector DB RAG and active audit context."""
    user_msg = req.message
    target_id = req.invoice_id

    # Build active invoice context
    inv_context = "No active invoice selected."
    if target_id and target_id in audit_cache:
        cached = audit_cache[target_id]
        inv_context = (
            f"Invoice ID: {cached.invoice_id}\n"
            f"Decision: {cached.decision.value.upper()}\n"
            f"Risk Score: {cached.risk_score}/100\n"
            f"Financial Exposure: ${cached.financial_exposure}\n"
            f"Discrepancies ({len(cached.reasons)}):\n"
            + "\n".join([f"- [{r.code.value}] {r.message}" for r in cached.reasons])
        )

    # Retrieve relevant clauses from Vector DB
    retrieved_clauses = contract_rag.retrieve(query=user_msg, top_k=2)
    contract_context = "\n".join([
        f"• Clause {c.clause_id} ('{c.title}'): '{c.content}'"
        for c, score in retrieved_clauses
    ])

    # Call LLM client (with automatic lowercase semantic fallback)
    reply = llm_client.chat_reply(
        query=user_msg,
        invoice_context=inv_context,
        contract_context=contract_context,
    )

    return {"reply": reply}


@app.get("/api/history")
def get_audit_history():
    """Retrieve audit history and recorded decisions."""
    return hitl_manager.list_recent_decisions()


# Mount static directory for frontend
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
