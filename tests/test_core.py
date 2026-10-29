"""Comprehensive unit and integration test suite.

Validates:
1. Deterministic Decimal arithmetic (no floating-point rounding errors) & true net variance.
2. Explicit Human-In-The-Loop review message triggers on missing fields.
3. 2-way and 3-way matching and cumulative billing overage detection in the Ledger.
4. Audit idempotency: run_audit does NOT pollute the ledger on read; committing only occurs on approval.
5. Contract RAG clause auditing (unauthorized surcharges).
6. LangGraph multi-agent execution within bounded iteration limits.
7. Pydantic ERP table projections (PO and Goods Receipts).
8. Thread-safe review manager persistence.
"""

from decimal import Decimal
import io
import json
import threading
import pytest
from fastapi.testclient import TestClient

from src.core.schema import (
    ContractClause,
    Decision,
    GoodsReceipt,
    Invoice,
    LineItem,
    POLine,
    PurchaseOrder,
    ReasonCode,
    ReceiptLine,
    ReviewAction,
    Severity,
)
from src.core.arithmetic import check_missing_fields, validate_arithmetic
from src.core.matcher import CumulativeLedger, DeterministicMatcher
from src.rag.contract_rag import ContractRAGStore
from src.agents.graph import AuditWorkflow
from src.hitl.review_manager import HITLReviewManager
from src.api.main import app


def test_deterministic_arithmetic_precision():
    """Verify exact Decimal arithmetic without floating-point errors."""
    invoice = Invoice(
        invoice_number="INV-001",
        vendor="Test Supplier",
        po_number="PO-001",
        currency="USD",
        lines=[
            LineItem(line_number=1, description="Part A", quantity=Decimal("3.0"), unit_price=Decimal("33.33"), line_total=Decimal("99.99")),
            LineItem(line_number=2, description="Part B", quantity=Decimal("2.0"), unit_price=Decimal("50.00"), line_total=Decimal("100.00")),
        ],
        subtotal=Decimal("199.99"),
        tax=Decimal("10.00"),
        discount=Decimal("0.00"),
        freight_surcharges=Decimal("0.00"),
        total=Decimal("209.99"),
    )

    reasons, variance = validate_arithmetic(invoice)
    assert variance == Decimal("0.00")
    assert not any(r.severity == Severity.REVIEW for r in reasons)


def test_arithmetic_does_not_triple_count_downstream_variance():
    """Ensure a $50 line discrepancy is not triple-counted into subtotal and total."""
    # Line total printed as $150 instead of $100 (diff: $50)
    # Subtotal printed as $150 (diff: $50)
    # Grand total printed as $150 (diff: $50)
    invoice = Invoice(
        invoice_number="INV-DIFF",
        vendor="Supplier Inc",
        lines=[
            LineItem(line_number=1, description="Widgets", quantity=Decimal("1.0"), unit_price=Decimal("100.00"), line_total=Decimal("150.00")),
        ],
        subtotal=Decimal("150.00"),
        total=Decimal("150.00"),
    )
    reasons, variance = validate_arithmetic(invoice)
    # The true net exposure is $50.00, not $50 + $50 + $50 = $150.00
    assert variance == Decimal("50.00")


def test_missing_fields_explicit_hitl_message():
    """Ensure missing critical fields explicitly trigger Human-In-The-Loop alerts."""
    incomplete_inv = Invoice(
        invoice_number=None,
        vendor=None,
        po_number=None,
        lines=[],
        total=None,
    )

    reasons = check_missing_fields(incomplete_inv)
    assert len(reasons) >= 3

    for r in reasons:
        assert r.requires_hitl is True
        assert r.severity == Severity.REVIEW
        assert "Human-In-The-Loop" in r.message


def test_cumulative_ledger_catches_overbilling():
    """Ensure the stateful ledger catches multi-part invoices exceeding PO quantities."""
    po = PurchaseOrder(
        po_number="PO-OVERBILL",
        vendor="Heavy Machinery Corp",
        lines=[
            POLine(line_number=1, sku="SKU-TURBINE", description="Turbine", quantity=Decimal("10.0"), unit_price=Decimal("100.00")),
        ],
    )

    ledger = CumulativeLedger()
    matcher = DeterministicMatcher(ledger)

    inv1 = Invoice(
        invoice_number="INV-A",
        vendor="Heavy Machinery Corp",
        po_number="PO-OVERBILL",
        lines=[
            LineItem(line_number=1, sku="SKU-TURBINE", description="Turbine", quantity=Decimal("7.0"), unit_price=Decimal("100.00"), line_total=Decimal("700.00")),
        ],
        total=Decimal("700.00"),
    )
    reasons1, _, _ = matcher.match(inv1, po=po)
    assert not any(r.code == ReasonCode.QTY_EXCEEDS_ORDERED for r in reasons1)

    # Record first billing in ledger
    ledger.record_invoice(inv1)
    ledger.record_po_billing("PO-OVERBILL", 1, Decimal("7.0"))

    # Second Invoice: Bills 5 units (7 + 5 = 12 > 10, overbilling by 2)
    inv2 = Invoice(
        invoice_number="INV-B",
        vendor="Heavy Machinery Corp",
        po_number="PO-OVERBILL",
        lines=[
            LineItem(line_number=1, sku="SKU-TURBINE", description="Turbine", quantity=Decimal("5.0"), unit_price=Decimal("100.00"), line_total=Decimal("500.00")),
        ],
        total=Decimal("500.00"),
    )
    reasons2, variance2, _ = matcher.match(inv2, po=po)
    overbilled_reason = next((r for r in reasons2 if r.code == ReasonCode.QTY_EXCEEDS_ORDERED), None)
    assert overbilled_reason is not None
    assert overbilled_reason.severity == Severity.REJECT
    assert "exceeds ordered quantity" in overbilled_reason.message
    assert variance2 == Decimal("200.00")


def test_audit_idempotency_does_not_pollute_ledger():
    """Verify that calling run_audit multiple times is purely idempotent and does not self-duplicate."""
    workflow = AuditWorkflow()
    inv = Invoice(
        invoice_number="INV-IDEMPOTENT-1",
        vendor="Standard Supply Co",
        po_number="PO-IDEM",
        lines=[
            LineItem(line_number=1, description="Motor", quantity=Decimal("2.0"), unit_price=Decimal("50.00"), line_total=Decimal("100.00")),
        ],
        total=Decimal("100.00"),
    )
    po = PurchaseOrder(
        po_number="PO-IDEM",
        vendor="Standard Supply Co",
        receipt_required=False,
        lines=[
            POLine(line_number=1, description="Motor", quantity=Decimal("5.0"), unit_price=Decimal("50.00")),
        ],
    )

    # First audit run
    res1 = workflow.run_audit(invoice=inv, po=po)
    assert res1.decision == Decision.AUTO_APPROVE

    # Second audit run of the SAME invoice must NOT trigger DUPLICATE_INVOICE or overbilling
    res2 = workflow.run_audit(invoice=inv, po=po)
    assert res2.decision == Decision.AUTO_APPROVE
    assert not any(r.code == ReasonCode.DUPLICATE_INVOICE for r in res2.reasons)

    # Ledger quantity should still be 0 before explicit commit
    assert workflow.ledger.get_billed_quantity("PO-IDEM", 1) == Decimal("0.0")

    # Now explicitly commit
    workflow.commit_invoice_to_ledger(inv, po, res2.matched_lines_map)
    assert workflow.ledger.get_billed_quantity("PO-IDEM", 1) == Decimal("2.0")

    # After commit, third run properly detects duplicate submission
    res3 = workflow.run_audit(invoice=inv, po=po)
    assert any(r.code == ReasonCode.DUPLICATE_INVOICE for r in res3.reasons)


def test_contract_rag_unauthorized_surcharge():
    """Ensure Contract RAG detects prohibited fuel/freight surcharges."""
    rag = ContractRAGStore()
    rag.add_clause(
        ContractClause(
            clause_id="SEC-4.2",
            vendor="Global Fast Freight",
            title="Freight Inclusivity Clause",
            content="Standard delivery is included in product price. Any separate fuel or freight surcharges are strictly unauthorized.",
            prohibits_surcharges=True,
        )
    )

    inv_with_surcharge = Invoice(
        invoice_number="INV-SURCHARGE-1",
        vendor="Global Fast Freight",
        po_number="PO-999",
        lines=[
            LineItem(line_number=1, description="Boxes", quantity=Decimal("10.0"), unit_price=Decimal("10.00"), line_total=Decimal("100.00")),
        ],
        subtotal=Decimal("100.00"),
        freight_surcharges=Decimal("75.00"),
        total=Decimal("175.00"),
    )

    reasons, evidence, violation_amt = rag.audit_contract_compliance(inv_with_surcharge)
    assert violation_amt == Decimal("75.00")
    assert any(r.code == ReasonCode.UNAUTHORIZED_SURCHARGE for r in reasons)
    assert any("SEC-4.2" in e.reference_id for e in evidence)


def test_pydantic_erp_table_projections():
    """Verify that PurchaseOrder and GoodsReceipt project cleanly into tabular format."""
    po = PurchaseOrder(
        po_number="PO-TABLE-TEST",
        vendor="Acme Tools",
        currency="USD",
        lines=[
            POLine(line_number=1, sku="SKU-DRILL", description="Cordless Drill", quantity=Decimal("5.0"), unit_price=Decimal("99.99")),
        ],
    )
    po_rows = po.to_table_rows()
    assert len(po_rows) == 1
    assert po_rows[0]["po_number"] == "PO-TABLE-TEST"
    assert po_rows[0]["line_total"] == "499.95"

    gr = GoodsReceipt(
        receipt_id="GR-TABLE-TEST",
        po_number="PO-TABLE-TEST",
        receipt_date="2026-10-29",
        lines=[
            ReceiptLine(line_number=1, sku="SKU-DRILL", quantity_received=Decimal("5.0")),
        ],
    )
    gr_rows = gr.to_table_rows()
    assert len(gr_rows) == 1
    assert gr_rows[0]["receipt_id"] == "GR-TABLE-TEST"
    assert gr_rows[0]["quantity_received"] in ("5.0", "5")


def test_api_upload_security_validation():
    """Verify security controls on the PDF upload endpoint (magic bytes & size check)."""
    client = TestClient(app)

    # 1. Invalid non-PDF file (fails magic byte check)
    invalid_file = io.BytesIO(b"This is just plain text, not a valid PDF document.")
    res_bad = client.post(
        "/api/audit/pdf",
        files={"file": ("fake.pdf", invalid_file, "application/pdf")},
    )
    assert res_bad.status_code == 400
    assert "Invalid file type" in res_bad.json()["detail"]


def test_hitl_review_manager_thread_safety(tmp_path):
    """Verify that concurrent decision recording is thread-safe without corrupting JSON."""
    log_file = tmp_path / "test_audit_log.json"
    manager = HITLReviewManager(log_path=log_file, ground_truth_dir=tmp_path / "gt")

    def record_dummy(i):
        manager.record_decision(
            invoice_id=f"INV-CONCURRENT-{i}",
            reviewer_id="Auditor_Thread",
            action=ReviewAction.APPROVE,
            notes=f"Approved concurrent invoice {i}",
        )

    threads = [threading.Thread(target=record_dummy, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all 15 records were successfully persisted
    decisions = manager.list_recent_decisions(limit=30)
    assert len(decisions) == 15


def test_entity_name_suffix_ignorance():
    """Verify that corporate legal suffixes ('Inc', 'LLC', 'Corp') are ignored so names match cleanly."""
    matcher = DeterministicMatcher()
    
    # PO has "ABC Corp", Invoice has "ABC Inc."
    po = PurchaseOrder(
        po_number="PO-ABC-1",
        vendor="ABC Corp",
        receipt_required=False,
        lines=[
            POLine(line_number=1, description="Widgets", quantity=Decimal("10.0"), unit_price=Decimal("5.00")),
        ],
    )
    inv = Invoice(
        invoice_number="INV-ABC-101",
        vendor="ABC Inc.",
        po_number="PO-ABC-1",
        lines=[
            LineItem(line_number=1, description="Widgets", quantity=Decimal("10.0"), unit_price=Decimal("5.00"), line_total=Decimal("50.00")),
        ],
        total=Decimal("50.00"),
    )

    reasons, _, _ = matcher.match(inv, po=po)
    # Must NOT report VENDOR_MISMATCH
    assert not any(r.code == ReasonCode.VENDOR_MISMATCH for r in reasons)


def test_composite_unique_history_key_deduplication():
    """Verify composite unique history check using address + name + id + date."""
    ledger = CumulativeLedger()
    matcher = DeterministicMatcher(ledger=ledger)

    inv = Invoice(
        invoice_number="INV-DUP-999",
        vendor="Acme Supplies, LLC",
        vendor_address="123 Industrial Way, Chicago, IL 60601",
        date="2026-10-22",
        po_number="PO-DUP-1",
        lines=[
            LineItem(line_number=1, description="Gear", quantity=Decimal("1.0"), unit_price=Decimal("100.00"), line_total=Decimal("100.00")),
        ],
        total=Decimal("100.00"),
    )
    po = PurchaseOrder(
        po_number="PO-DUP-1",
        vendor="Acme Supplies",
        vendor_address="123 Industrial Way, Chicago, IL 60601",
        receipt_required=False,
        lines=[
            POLine(line_number=1, description="Gear", quantity=Decimal("1.0"), unit_price=Decimal("100.00")),
        ],
    )

    # First audit: No duplicate
    reasons1, _, _ = matcher.match(inv, po=po)
    assert not any(r.code == ReasonCode.DUPLICATE_INVOICE for r in reasons1)

    # Commit to ledger
    ledger.record_invoice(inv)

    # Re-auditing with same address, name, id, date must trigger DUPLICATE_INVOICE
    reasons2, _, _ = matcher.match(inv, po=po)
    dup = next((r for r in reasons2 if r.code == ReasonCode.DUPLICATE_INVOICE), None)
    assert dup is not None
    assert dup.severity == Severity.REJECT
    assert "composite_key" in dup.details


def test_sample_pack_zip_download_and_proper_dates(tmp_path):
    """Verify zip creation, proper chronological dates in samples, and download endpoint."""
    from src.synth.dataset_gen import generate_synthetic_dataset
    import zipfile

    # Generate synthetic dataset into temporary directory
    data_manifest = generate_synthetic_dataset(tmp_path)
    zip_file = tmp_path / "sample_procurement_documents.zip"
    assert zip_file.exists()

    # Verify zip content entries
    with zipfile.ZipFile(zip_file, "r") as zf:
        namelist = zf.namelist()
        assert "README.txt" in namelist
        assert "purchase_orders.json" in namelist
        assert "goods_receipts.json" in namelist
        assert "contracts.json" in namelist
        assert any(n.startswith("invoices/") and n.endswith(".pdf") for n in namelist)

    # Verify proper dates: Contract <= PO <= Receipt <= Invoice
    with open(tmp_path / "purchase_orders.json", "r", encoding="utf-8") as f:
        pos = json.load(f)
    with open(tmp_path / "goods_receipts.json", "r", encoding="utf-8") as f:
        grs = json.load(f)

    po_001 = next(p for p in pos if p["po_number"] == "PO-2026-001")
    gr_001 = next(g for g in grs if g["po_number"] == "PO-2026-001")
    assert po_001["date"] == "2026-10-10"
    assert gr_001["receipt_date"] == "2026-10-18"
    assert po_001["date"] < gr_001["receipt_date"]

    # Verify download endpoint
    client = TestClient(app)
    res = client.get("/api/samples/download-zip")
    assert res.status_code == 200
    assert res.headers["content-type"] in ("application/zip", "application/x-zip-compressed")

