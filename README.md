# ProcureGuard: Multi-Agent Procurement & Contract Auditor

An enterprise procurement intelligence and audit system combining **100% deterministic arithmetic & stateful ledger tracking** with **FastAPI, Vector RAG, and LangGraph-powered Multi-Agents**.

ProcureGuard protects accounts payable and procurement teams against:
- Invoice arithmetic discrepancies & rounding errors
- Purchase Order (PO) rate card variances and quantity overbilling
- Unreceived shipments (3-way matching against warehouse Good Orders / Goods Receipts)
- Hidden stealth charges (e.g., unauthorized fuel or freight surcharges prohibited by Master Service Agreements)
- Multi-invoice cumulative overbilling across split shipments
- Missing mandatory fields, with explicit routing to **Human-in-the-Loop (HITL)** review

---

## 🏛️ Core Architectural Principle

> **"Never let an LLM do arithmetic; never let a rule engine interpret legal nuances."**
>
> Mathematical reconciliation, tax calculations, and cumulative volume limits remain **100% deterministic and auditable** using fixed-precision decimal arithmetic (`decimal.Decimal`). GenAI and Multi-Agent workflows are utilized where they excel: dynamically extracting unstructured documents, interpreting complex contractual clauses, correlating anomalies, explaining discrepancies, and conducting conversational investigations.

---

## 🤖 Multi-Agent Architecture & Specialized Sub-Agents

ProcureGuard organizes the audit lifecycle into specialized, bounded sub-agents coordinated via a **LangGraph** state machine. Each sub-agent has a dedicated responsibility and specific deterministic or AI-powered tools:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LANGGRAPH AUDIT ORCHESTRATION                                   │
└─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
    ┌───────────────────────────┐                                     ┌───────────────────────────┐
    │ 1. Validator Sub-Agent    │                                     │ Dynamic Ingestion Tools   │
    │    (Arithmetic & Schema)  │                                     │ • PyMuPDF Spatial Parser  │
    └─────────────┬─────────────┘                                     │ • Schema-Constrained LLM  │
                  │                                                   │ • Dynamic Table Parser    │
                  ▼                                                   └───────────────────────────┘
    ┌───────────────────────────┐
    │ 2. 3-Way Reconciler Agent │
    │    (ERP & Warehouse Proof)│
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │ 3. Compliance & Triage    │◄────────────► ┌───────────────────────────┐
    │    Sub-Agent (Vector RAG) │               │ 5. AI Copilot Sub-Agent   │
    └─────────────┬─────────────┘               │    (Conversational Chat)  │
                  │                             └───────────────────────────┘
                  ▼
    ┌───────────────────────────┐
    │ 4. Human-in-the-Loop      │
    │    Reviewer Cockpit       │
    └───────────────────────────┘
```

### 1. Ingestion & Preprocessing Sub-Agent
- **Role:** Digitizes incoming documents regardless of layout, format, or non-standard file naming.
- **Tools:**
  - **PyMuPDF Spatial Parser:** Extracts raw text, spatial coordinates, and tabular bounding boxes.
  - **Schema-Constrained LLM Extractor:** Uses Pydantic structured extraction schemas (`InvoiceExtractionSchema`) with zero-shot layout understanding and local semantic fallback.
  - **Dynamic Table Parser (`table_parser.py`):** Dynamically detects and parses tabular Purchase Orders and Good Orders (Goods Receipts) from CSV, TSV, or JSON tables independent of file names using fuzzy header synonym mapping (`_find_field`).

### 2. Validator Sub-Agent (`_validator_node`)
- **Role:** Verifies structural completeness and enforces mathematical integrity.
- **Tools:**
  - **Deterministic Decimal Arithmetic Engine (`arithmetic.py`):** Uses Python's `decimal.Decimal` with fixed-precision rounding (`ROUND_HALF_UP`). Recomputes `quantity × unit_price` per line, validates `subtotal = Σ line_totals`, and verifies `total = subtotal + tax + freight - discount`.
  - **Missing Required Fields Checker:** Verifies presence of business-critical fields (`invoice_number`, `vendor`, `po_number`, `lines`).
- **Behavior:**
  - Flags explicit Human-in-the-Loop review if mandatory fields are missing from scanned or incomplete documents.
  - Detects arithmetic variances without cascading/triple-counting discrepancies.

### 3. 3-Way Reconciliation Sub-Agent (`_reconciler_node`)
- **Role:** Multi-document operational verification across Invoices, Purchase Orders, and Warehouse Goods Receipts.
- **Tools:**
  - **Entity Normalization Tool (`normalization.py`):** Normalizes vendor names by stripping corporate suffixes (`Inc`, `LLC`, `Corp`, `GmbH`, `PLC`, etc.) and standardizing street addresses to avoid false mismatch rejections.
  - **Deterministic 3-Way Matcher (`matcher.py`):** Matches line items against authorized PO rate cards, verifies quantities against physical goods receipt delivery dates, and detects open/unfulfilled lines.
  - **Composite Deduplication Tool:** Uses a 4-factor composite hash: `Hash(Vendor + Invoice_ID + Date + Address)` to catch accidental or fraudulent duplicate submissions.
  - **Stateful Cumulative Ledger (`matcher.py`):** Maintains historical line-item commitments across split shipments to catch cumulative overbilling.

### 4. Legal Compliance Sub-Agent (`_contract_and_triage_node`)
- **Role:** Validates invoice line items against contractual terms and Master Service Agreements (MSAs).
- **Tools:**
  - **Dedicated Contract Clause Parser (`table_parser.py`):** Dynamically extracts clauses from CSV tables, JSON arrays, or raw text/markdown agreements.
  - **Contract Vector RAG Store (`contract_rag.py`):** Hybrid BM25 keyword matching + semantic cosine vector search over indexed agreement clauses.
  - **Clause Compliance Analyzer:** Analyzes line items against contract rules to identify unauthorized stealth surcharges (fuel fees, handling fees, shipping markups) and unapplied volume discounts.

### 5. Triage & Decision Synthesis Sub-Agent
- **Role:** Synthesizes findings into actionable outcomes and computes net financial exposure.
- **Tools:**
  - **Risk Score Evaluator:** Computes a normalized risk score (0–100) based on discrepancy severity (`INFO`, `REVIEW`, `REJECT`).
  - **Evidence Bundle Aggregator:** Builds an audit-ready chain of custody attaching citations (PO lines, receipt packing slips, clause excerpts).
  - **Workflow Decision Router:** Recommends `AUTO_APPROVE` (clean matches), `NEEDS_REVIEW` (variances / incomplete fields), or `REJECT` (contract violations / hard overbilling).

### 6. AI Copilot Chatbot Sub-Agent (`auditor_chatbot`)
- **Role:** Interactive real-time assistant for human procurement auditors.
- **Tools:**
  - **Vector RAG Clause Retriever:** Pulls contract clauses on demand based on auditor prompts.
  - **Session Context Injector:** Dynamically grounds replies in the active invoice findings, calculated financial exposure, and ERP table lines.
  - **Session Lifecycle Controller:** Automatically restarts conversation history for each isolated audit session.

---

## 🔍 Why Use a Stateful Ledger?

In real-world enterprise procurement, **auditing invoices in isolation (statelessly) causes multi-million dollar overpayments**.

1. **Cumulative Overbilling Across Split Shipments:**
   A company issues a PO for 10 units. A vendor ships in two batches:
   - *Week 1:* Invoice A bills **7 units** $\rightarrow$ Valid ($7 \le 10$) $\rightarrow$ Paid.
   - *Week 3:* Invoice B bills **5 units** $\rightarrow$ If audited statelessly, an LLM sees $5 \le 10$ and approves it!
   - **With the Cumulative Ledger:** The ledger tracks prior billings: $7 + 5 = 12 > 10$. It automatically flags **`QTY_EXCEEDS_ORDERED`** (overbilled by 2 units).
2. **Duplicate Invoice Submission Prevention:**
   Vendors often re-submit invoices accidentally or fraudulently. The Ledger maintains unique submission fingerprints (`vendor`, `invoice_number`, `date`, `address`) to catch exact duplicates (`DUPLICATE_INVOICE`) and near-duplicates (`POSSIBLE_DUPLICATE`).
3. **Idempotency & Side-Effect Free Reads:**
   Running an audit check does not pollute the ledger. Commits only occur when human auditors explicitly approve the transaction.

---

## 📋 Tripartite Procurement Document Formats

The system models how enterprise ERP systems operate:

- **Invoices & Bills**: **PDF Documents** (native PDF invoices with visual coordinate text layers).
- **Purchase Orders (POs)**: **Tabular CSV Table** (`po_number, vendor, vendor_address, date, currency, line_number, sku, description, quantity, unit_price`).
- **Good Orders / Goods Receipts**: **Tabular CSV Table** (`receipt_id, po_number, receipt_date, line_number, sku, quantity_received`).
- **Contracts / MSAs**: **Vector DB Store** with hybrid keyword/vector search and dynamic clause creation.

---

## ⚡ Key Features

1. **Deterministic Arithmetic Precision**: Fixed-precision decimal arithmetic (`decimal.Decimal`) eliminates floating-point rounding errors.
2. **Dynamic Name-Independent Ingestion**: Ingests PO and Good Orders tables in CSV, TSV, or JSON format regardless of input file names.
3. **Session Folder Persistence**: Each audit run operates in an isolated folder (`data/sessions/{session_id}/`), maintaining a single copy of files without redundant storage.
4. **File Cap & Batch Limits**: Protects server resources with configurable caps (up to 25 invoices per batch, 25 MB per file).
5. **Parallel Step-by-Step Stepper**: Concurrently executes contract chunking, visual PDF extraction, 3-way reconciliation, and stateful ledger tracking.
6. **Explicit Human-in-the-Loop (HITL) Guidance**: Missing required fields or financial exposure variances route to human review with actionable dispute guidance.
7. **Clean Onboarding Portal**: Zero pre-loaded tables or stale documents on startup; starts clean and activates upon document selection.
8. **Consolidated Benchmark Samples**: Complete sample pack organized under `data/sample/` with downloadable ZIP archive.

---

## 🚀 Quickstart Guide

### 1. Configure Environment (Optional)
To enable cloud LLM reasoning, create a `.env` file:
```bash
GEMINI_API_KEY="your-gemini-api-key"
# or
OPENAI_API_KEY="your-openai-api-key"
```
*(If no API key is provided, the local heuristic and semantic extraction engine runs automatically without crashing).*

### 2. Run the Automated Test Suite
```bash
python -u -m pytest -p no:deepeval tests/test_core.py -v
```

### 3. Start the Server
```bash
python run_server.py
```
- **Web Cockpit & Copilot:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🖥️ Cockpit Interface Views

1. **Invoice Audit Cockpit**:
   - Clean drag-and-drop file upload portal with live file cap validation.
   - Real-time parallel stepper animation displaying execution across multi-agent steps.
   - Discrepancy cards with severity badges (`AUTO_APPROVE`, `NEEDS_REVIEW`, `REJECT`).
   - Attached evidence bundle citing specific PO lines and contract clauses.
   - Human-in-the-Loop decision bar (`Approve`, `Flag for Correction`, `Reject`).
   - Embedded AI Copilot Chatbot restarted per session.
2. **ERP Tables (PO & Receipts)**:
   - Interactive relational tables for authorized Purchase Orders and Good Orders / Goods Receipts.
3. **Vector DB Contracts**:
   - Visual inspector of indexed contract clauses, token counts, and surcharge restrictions.
   - Form to add and tokenize new contract clauses in real time.

---

## 🧪 Benchmark Scenarios (`data/sample/`)

| Scenario ID | Test Condition | Expected Action | Reason Code |
| :--- | :--- | :--- | :--- |
| `INV-101` | Clean 3-way match against PO-2026-001 and Goods Receipt | `AUTO_APPROVE` | Zero discrepancies |
| `INV-102` | Unit price inflated from $600 to $750 | `NEEDS_REVIEW` | `PRICE_VARIANCE` |
| `INV-103` | Omitted vendor and line items from document | `NEEDS_REVIEW` | `MISSING_REQUIRED_FIELD` (Explicit HITL) |
| `INV-105` | $150 freight surcharge added, violating MSA clause | `REJECT` | `UNAUTHORIZED_SURCHARGE` |

---

## 📂 Project Structure

```
Intelligent_Invoice_System/
  ├── data/
  │   ├── sample/                    # Unified sample benchmark documents
  │   │   ├── invoices/              # Native PDF sample invoices
  │   │   ├── purchase_orders.csv    # ERP PO table in CSV format
  │   │   ├── goods_receipts.csv     # Warehouse Good Orders table in CSV format
  │   │   ├── contracts.json         # Contract agreement clauses
  │   │   ├── ground_truth/          # Benchmark evaluation labels
  │   │   └── sample_procurement_documents.zip # Downloadable sample pack
  │   └── sessions/                  # Isolated runtime session workspaces
  ├── src/
  │   ├── core/
  │   │   ├── schema.py              # Strict Decimal types, Pydantic domain models
  │   │   ├── arithmetic.py          # Deterministic line item, subtotal & total validation
  │   │   ├── matcher.py             # 2-way/3-way reconciliation + stateful cumulative ledger
  │   │   ├── table_parser.py        # Dynamic name-independent CSV/TSV table & clause parser
  │   │   ├── normalization.py       # Corporate entity & address normalization
  │   │   ├── session_manager.py     # Isolated session workspace & parallel execution
  │   │   └── llm.py                 # Dynamic LLM client (Gemini/OpenAI/Fallback)
  │   ├── extractor/
  │   │   └── pdf_extractor.py       # Dynamic PDF extractor with spatial coordinate reconstruction
  │   ├── rag/
  │   │   └── contract_rag.py        # Hybrid BM25 & vector RAG for legal compliance
  │   ├── agents/
  │   │   └── graph.py               # Streamlined LangGraph StateGraph workflow & sub-agents
  │   ├── hitl/
  │   │   └── review_manager.py      # Thread-safe reviewer recording & audit trail logging
  │   ├── synth/
  │   │   └── dataset_gen.py         # Chronological synthetic dataset generator
  │   └── api/
  │       └── main.py                # FastAPI REST service & session endpoints
  ├── static/
  │   ├── index.html                 # Multi-tab Cockpit UI with clean upload portal
  │   └── app.js                     # Client controller, file caps, and stepper manager
  ├── tests/
  │   └── test_core.py               # Automated test suite (17/17 tests passing)
  ├── run_server.py                  # Server launcher
  └── README.md
```
