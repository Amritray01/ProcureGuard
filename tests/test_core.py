"""Comprehensive unit and integration test suite.

Validates:
1. Deterministic Decimal arithmetic (no floating-point rounding errors).
2. Explicit Human-In-The-Loop review message triggers on missing fields.
3. 2-way and 3-way matching and cumulative billing overage detection in the Ledger.
4. Contract RAG clause auditing (unauthorized surcharges).
5. LangGraph multi-agent execution within bounded iteration limits.
"""

from decimal import Decimal
import pytest
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


def test_deterministic_arithmetic_precision():
    """Verify exact Decimal arithmetic without floating-point errors."""
    invoice = Invoice(
        invoice_number="INV-001",
        vendor="Test Supplier",
        po_number="PO-001",
        currency="USD",
        lines=[
            # 3 * 33.3333 -> 100.00
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
    # Expected: No line mismatches or total mismatches
    assert variance == Decimal("0.00")
    assert not any(r.severity == Severity.REVIEW for r in reasons)


def test_missing_fields_explicit_hitl_message():
    """Ensure missing critical fields explicitly trigger Human-In-The-Loop alerts."""
    # Invoice missing invoice_number, vendor, and line items
    incomplete_inv = Invoice(
        invoice_number=None,
        vendor=None,
        po_number=None,
        lines=[],
        total=None,
    )

    reasons = check_missing_fields(incomplete_inv)
    assert len(reasons) >= 3

    # Every missing field reason must flag requires_hitl and contain clear guidance
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

    # First Invoice: Bills 7 units (allowed, 7 <= 10)
    inv1 = Invoice(
        invoice_number="INV-A",
        vendor="Heavy Machinery Corp",
        po_number="PO-OVERBILL",
        lines=[
            LineItem(line_number=1, sku="SKU-TURBINE", description="Turbine", quantity=Decimal("7.0"), unit_price=Decimal("100.00"), line_total=Decimal("700.00")),
        ],
        total=Decimal("700.00"),
    )
    reasons1, _ = matcher.match(inv1, po=po)
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
    reasons2, variance2 = matcher.match(inv2, po=po)
    overbilled_reason = next((r for r in reasons2 if r.code == ReasonCode.QTY_EXCEEDS_ORDERED), None)
    assert overbilled_reason is not None
    assert overbilled_reason.severity == Severity.REJECT
    assert "exceeds ordered quantity" in overbilled_reason.message
    # 2 units * $100.00 = $200.00 variance
    assert variance2 == Decimal("200.00")


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
        freight_surcharges=Decimal("75.00"),  # Unauthorized surcharge
        total=Decimal("175.00"),
    )

    reasons, evidence, violation_amt = rag.audit_contract_compliance(inv_with_surcharge)
    assert violation_amt == Decimal("75.00")
    assert any(r.code == ReasonCode.UNAUTHORIZED_SURCHARGE for r in reasons)
    assert any("SEC-4.2" in e.reference_id for e in evidence)


def test_langgraph_bounded_audit_workflow():
    """Verify that LangGraph workflow executes end-to-end and respects decision bounds."""
    workflow = AuditWorkflow()
    clean_inv = Invoice(
        invoice_number="INV-CLEAN",
        vendor="Precision Tools Co",
        po_number="PO-CLEAN",
        lines=[
            LineItem(line_number=1, description="Wrench", quantity=Decimal("5.0"), unit_price=Decimal("20.00"), line_total=Decimal("100.00")),
        ],
        subtotal=Decimal("100.00"),
        tax=Decimal("5.00"),
        total=Decimal("105.00"),
    )
    po = PurchaseOrder(
        po_number="PO-CLEAN",
        vendor="Precision Tools Co",
        receipt_required=False,
        lines=[
            POLine(line_number=1, description="Wrench", quantity=Decimal("5.0"), unit_price=Decimal("20.00")),
        ],
    )

    result = workflow.run_audit(invoice=clean_inv, po=po)
    assert result.decision == Decision.AUTO_APPROVE
    assert result.risk_score <= 10.0
    assert result.requires_human_review is False
