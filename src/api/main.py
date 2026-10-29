"""FastAPI Web Service for Procurement Invoice & Contract Auditor.

Exposes REST endpoints for:
- Session-based isolated auditing with folder persistence in data/sessions/{session_id}/
- Step-by-step parallelized audit workflow:
    1. Parallel Contract Chunking & Vector DB Indexing
    2. Parallel Dynamic PDF Extraction & Horizontally Reconstructed Parsing
    3. Multi-Agent 3-Way Reconciliation & Vector RAG Contract Audit
    4. Cumulative Stateful Ledger & Composite History (Address+Name+ID+Date) Verification
- Tabular ERP Views (Purchase Orders & Goods Receipts) strictly scoped to active session
- Isolated Vector DB Contract Clauses inspection and querying
- Human-in-the-Loop review actions & Chatbot copilot
- Sample Document Pack download (.zip)
"""

import asyncio
from collections import OrderedDict
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from src.core.session_manager import AuditSession, SessionManager
from src.core.table_parser import (
    export_goods_receipts_to_csv,
    export_purchase_orders_to_csv,
    parse_contract_clauses,
    parse_goods_receipts,
    parse_purchase_orders,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB file cap
MAX_INVOICE_COUNT = 25  # Max 25 invoices per batch

app = FastAPI(
    title="Procurement Invoice & Contract Auditor API",
    description="Deterministic 3-Way Math + Contract Vector RAG + LangGraph Multi-Agents",
    version="2.0.0",
)

# Safe CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prevent aggressive browser caching of UI code and static assets
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".html", ".js", "/")) or request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Global Session Manager for isolated, persistent folder workspaces
session_manager = SessionManager(data_dir=DATA_DIR)


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


# --- Session-Based Multi-Agent Parallel Workflow ---

@app.post("/api/session/start")
async def start_session_workflow(
    invoices: List[UploadFile] = File(default=[]),
    purchase_orders: Optional[UploadFile] = File(default=None),
    goods_receipts: Optional[UploadFile] = File(default=None),
    contracts: Optional[UploadFile] = File(default=None),
    use_sample_pack: bool = Form(default=False),
):
    """Create isolated session and execute step-by-step parallelized workflow."""
    # 1. Initialize new session workspace
    session = session_manager.create_session()

    # Enforce invoice count file cap
    if len(invoices) > MAX_INVOICE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum invoice file cap exceeded ({len(invoices)} files provided). Limit is {MAX_INVOICE_COUNT} invoices per session.",
        )

    invoice_files: List[Tuple[str, bytes]] = []
    contracts_data: List[Dict[str, Any]] = []

    if use_sample_pack:
        # Load verified sample benchmark files into this session workspace
        if not (DATA_DIR / "purchase_orders.csv").exists() and not (DATA_DIR / "purchase_orders.json").exists():
            generate_synthetic_dataset(DATA_DIR)

        # Invoices (loaded directly into memory for isolated session persistence)
        sample_pdf_dir = DATA_DIR / "invoices"
        if sample_pdf_dir.exists():
            for p_file in sorted(sample_pdf_dir.glob("*.pdf")):
                invoice_files.append((p_file.name, p_file.read_bytes()))

        # Orders strictly loaded from CSV file format for sample docs
        po_csv = DATA_DIR / "purchase_orders.csv"
        if po_csv.exists():
            session.purchase_orders = parse_purchase_orders(po_csv.read_bytes())
        elif (DATA_DIR / "purchase_orders.json").exists():
            session.purchase_orders = parse_purchase_orders((DATA_DIR / "purchase_orders.json").read_bytes())

        # Good Orders / Receipts strictly loaded from CSV file format for sample docs
        gr_csv = DATA_DIR / "goods_receipts.csv"
        if gr_csv.exists():
            session.goods_receipts = parse_goods_receipts(gr_csv.read_bytes())
        elif (DATA_DIR / "goods_receipts.json").exists():
            session.goods_receipts = parse_goods_receipts((DATA_DIR / "goods_receipts.json").read_bytes())

        # Contracts loaded dynamically
        if (DATA_DIR / "contracts.json").exists():
            c_raw = (DATA_DIR / "contracts.json").read_bytes()
            contracts_data = [c.model_dump() for c in parse_contract_clauses(c_raw)]

    else:
        # Process user-uploaded tabular and document files dynamically (name-independent)
        if purchase_orders:
            po_bytes = await purchase_orders.read()
            if len(po_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Purchase Orders file exceeds 25 MB cap.")
            try:
                session.purchase_orders = parse_purchase_orders(po_bytes, filename=purchase_orders.filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid PO table format: {str(e)}")

        if goods_receipts:
            gr_bytes = await goods_receipts.read()
            if len(gr_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Goods Receipts file exceeds 25 MB cap.")
            try:
                session.goods_receipts = parse_goods_receipts(gr_bytes, filename=goods_receipts.filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid Good Orders / Receipt table format: {str(e)}")

        if contracts:
            c_bytes = await contracts.read()
            if len(c_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Contracts file exceeds 25 MB cap.")
            try:
                clauses = parse_contract_clauses(c_bytes, filename=contracts.filename)
                contracts_data = [c.model_dump() for c in clauses]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse contract clauses: {str(e)}")

        for inv_file in invoices:
            content = await inv_file.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail=f"File {inv_file.filename} exceeds 25 MB limit.")
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail=f"File {inv_file.filename} is not a valid PDF document.")
            fname = Path(inv_file.filename or "invoice.pdf").name
            # Store only one copy: directly in session invoices folder during extraction
            invoice_files.append((fname, content))

    if not invoice_files:
        raise HTTPException(status_code=400, detail="No invoice documents provided for auditing.")

    # Save session POs and Receipts to session directory (both tabular CSV and JSON)
    po_list = list(session.purchase_orders.values())
    gr_list = list(session.goods_receipts.values())

    with open(session.session_dir / "purchase_orders.csv", "w", encoding="utf-8") as f:
        f.write(export_purchase_orders_to_csv(po_list))
    with open(session.session_dir / "purchase_orders.json", "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in po_list], f, indent=2)

    with open(session.session_dir / "goods_receipts.csv", "w", encoding="utf-8") as f:
        f.write(export_goods_receipts_to_csv(gr_list))
    with open(session.session_dir / "goods_receipts.json", "w", encoding="utf-8") as f:
        json.dump([g.model_dump(mode="json") for g in gr_list], f, indent=2)

    # --- Step 1: Parallel Contract Chunking & Vector DB Indexing ---
    await session.step_chunk_contracts_parallel(contracts_data)

    # --- Step 2: Parallel Dynamic PDF Extraction ---
    await session.step_extract_invoices_parallel(invoice_files)

    # --- Step 3 & 4: Multi-Agent 3-Way Reconciliation & Cumulative Ledger ---
    await session.step_audit_and_ledger_process()

    invoice_summaries = []
    for inv_id, inv in session.invoices.items():
        res = session.audit_results.get(inv_id)
        if res:
            invoice_summaries.append({
                "id": inv_id,
                "vendor": inv.vendor or "Unknown Vendor",
                "po_number": inv.po_number or "N/A",
                "date": inv.date or "N/A",
                "decision": res.decision.value,
                "risk_score": res.risk_score,
                "financial_exposure": str(res.financial_exposure),
            })

    return {
        "status": "completed",
        "session_id": session.session_id,
        "steps": session.steps,
        "invoices": invoice_summaries,
        "results": {
            inv_id: res.model_dump(mode="json")
            for inv_id, res in session.audit_results.items()
        },
    }


@app.get("/api/session/current")
def get_current_session():
    """Return active session identifier and step status."""
    session = session_manager.get_session()
    if not session:
        return {"session_id": None, "status": "idle"}
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "status": session.status,
        "invoice_count": len(session.invoices),
        "steps": session.steps,
    }


@app.post("/api/session/reset")
def reset_current_session():
    """Clear active session and reset to initial clean upload state."""
    session_manager.current_session_id = None
    return {"status": "reset", "session_id": None}


@app.get("/api/sessions")
def list_persisted_sessions():
    """List all persisted sessions in data/sessions/."""
    return session_manager.list_sessions()


# --- Scoped ERP Tabular Endpoints (Only for Active Session) ---

@app.get("/api/erp/purchase-orders")
def get_purchase_orders_table(session_id: Optional[str] = None):
    """Return PO table rows strictly for the active session (empty if no session)."""
    session = session_manager.get_session(session_id)
    if not session:
        return []
    rows = []
    for po in session.purchase_orders.values():
        rows.extend(po.to_table_rows())
    return rows


@app.get("/api/erp/goods-receipts")
def get_goods_receipts_table(session_id: Optional[str] = None):
    """Return Goods Receipt table rows strictly for the active session (empty if no session)."""
    session = session_manager.get_session(session_id)
    if not session:
        return []
    rows = []
    for gr in session.goods_receipts.values():
        rows.extend(gr.to_table_rows())
    return rows


# --- Scoped Vector DB Endpoints ---

@app.get("/api/rag/clauses")
def list_vector_db_clauses(session_id: Optional[str] = None):
    """List contract clauses indexed in the active session Vector DB."""
    session = session_manager.get_session(session_id)
    if not session:
        return []
    return session.contract_rag.list_all_clauses()


@app.post("/api/rag/clauses")
def add_vector_db_clause(req: AddClauseRequest, session_id: Optional[str] = None):
    """Add a custom contract clause to active session's Vector DB."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please start an audit session first.")
    clause = ContractClause(
        clause_id=req.clause_id,
        vendor=req.vendor,
        title=req.title,
        content=req.content,
        category=req.category,
        prohibits_surcharges=req.prohibits_surcharges,
        discount_percentage=req.discount_percentage,
    )
    details = session.contract_rag.add_clause(clause)
    return {"status": "indexed", "clause_details": details}


# --- Human-in-the-Loop & Chatbot Endpoints ---

@app.post("/api/review")
def record_human_decision(req: ReviewRequest, session_id: Optional[str] = None):
    """Record Human-in-the-Loop decision in active session and commit to ledger if approved."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active audit session.")
    decision = session.record_human_decision(
        invoice_id=req.invoice_id,
        reviewer_id=req.reviewer_id,
        action=req.action.value,
        notes=req.notes,
        overridden_reasons=req.overridden_reasons,
    )
    return {"status": "recorded", "decision": decision.model_dump()}


@app.post("/api/chat")
async def auditor_chatbot(req: ChatRequest, session_id: Optional[str] = None):
    """LLM-backed conversational copilot with Vector DB RAG and active session audit context."""
    session = session_manager.get_session(session_id)
    inv_context = "No active invoice selected."
    contract_context = ""

    if session:
        if req.invoice_id and req.invoice_id in session.audit_results:
            cached = session.audit_results[req.invoice_id]
            inv_context = (
                f"Invoice ID: {cached.invoice_id}\n"
                f"Decision: {cached.decision.value.upper()}\n"
                f"Risk Score: {cached.risk_score}/100\n"
                f"Financial Exposure: ${cached.financial_exposure}\n"
                f"Discrepancies ({len(cached.reasons)}):\n"
                + "\n".join([f"- [{r.code.value}] {r.message}" for r in cached.reasons])
            )
        retrieved_clauses = session.contract_rag.retrieve(query=req.message, top_k=2)
        contract_context = "\n".join([
            f"• Clause {c.clause_id} ('{c.title}'): '{c.content}'"
            for c, score in retrieved_clauses
        ])

    reply = await asyncio.to_thread(
        llm_client.chat_reply,
        query=req.message,
        invoice_context=inv_context,
        contract_context=contract_context,
    )
    return {"reply": reply}


# --- Sample Pack Download ---

@app.get("/api/samples/download-zip")
def download_sample_documents_zip():
    """Download ZIP archive containing sample benchmark documents."""
    zip_path = DATA_DIR / "sample_procurement_documents.zip"
    if not zip_path.exists():
        generate_synthetic_dataset(DATA_DIR)
    return FileResponse(
        path=str(zip_path),
        filename="sample_procurement_documents.zip",
        media_type="application/zip",
    )


@app.get("/api/dataset/samples")
def get_sample_invoices():
    """List sample benchmark invoice labels."""
    manifest_file = DATA_DIR / "manifest.json"
    if not manifest_file.exists():
        generate_synthetic_dataset(DATA_DIR)
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Backward-compatible single PDF audit
@app.post("/api/audit/pdf")
async def audit_uploaded_pdf(file: UploadFile = File(...)):
    """Upload single PDF invoice into an active or new session workspace."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds limit of 25 MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only valid PDF documents with %PDF- header are accepted.")

    fname = Path(file.filename or "uploaded.pdf").name

    session = session_manager.get_session()
    if not session:
        session = session_manager.create_session()

    await session.step_extract_invoices_parallel([(fname, content)])
    await session.step_audit_and_ledger_process()

    if fname in session.audit_results:
        return session.audit_results[fname].model_dump(mode="json")
    for res in session.audit_results.values():
        return res.model_dump(mode="json")

    raise HTTPException(status_code=500, detail="Audit processing failed.")


@app.post("/api/audit/sample/{invoice_id}")
async def audit_sample_endpoint(invoice_id: str):
    """Run audit on a benchmark sample invoice and return result."""
    session = session_manager.get_session()
    if not session:
        session = session_manager.create_session()

    manifest_file = DATA_DIR / "manifest.json"
    if not manifest_file.exists():
        generate_synthetic_dataset(DATA_DIR)
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_inv = next((inv for inv in manifest["invoices"] if inv["id"] == invoice_id), None)
    if not target_inv:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found.")

    po_csv = DATA_DIR / "purchase_orders.csv"
    po_json = DATA_DIR / "purchase_orders.json"
    if not session.purchase_orders:
        if po_csv.exists():
            session.purchase_orders = parse_purchase_orders(po_csv.read_bytes())
        elif po_json.exists():
            session.purchase_orders = parse_purchase_orders(po_json.read_bytes())

    gr_csv = DATA_DIR / "goods_receipts.csv"
    gr_json = DATA_DIR / "goods_receipts.json"
    if not session.goods_receipts:
        if gr_csv.exists():
            session.goods_receipts = parse_goods_receipts(gr_csv.read_bytes())
        elif gr_json.exists():
            session.goods_receipts = parse_goods_receipts(gr_json.read_bytes())

    if not session.contract_rag.clauses and (DATA_DIR / "contracts.json").exists():
        with open(DATA_DIR / "contracts.json", "r", encoding="utf-8") as f:
            for c in json.load(f):
                session.contract_rag.add_clause(ContractClause(**c))

    pdf_path = DATA_DIR / "invoices" / target_inv["file"]
    content = pdf_path.read_bytes()

    await session.step_extract_invoices_parallel([(target_inv["file"], content)])
    await session.step_audit_and_ledger_process()

    res = session.audit_results.get(invoice_id)
    if res:
        return res.model_dump(mode="json")
    for r in session.audit_results.values():
        return r.model_dump(mode="json")
    raise HTTPException(status_code=500, detail="Audit failed.")


# Mount static directory for frontend
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
