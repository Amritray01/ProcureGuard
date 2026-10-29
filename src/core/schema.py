"""Strict data models and schemas for the Procurement Invoice & Contract Auditor.

Money and quantities are strictly represented using decimal.Decimal to eliminate
floating-point rounding errors and ensure 100% deterministic auditing.
All structured output and ERP tabular projections use explicit Pydantic Field descriptions.
"""

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator


def serialize_decimal(v: Decimal) -> str:
    """Format Decimal to 2 decimal places for clean JSON serialization."""
    return f"{v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


Money = Annotated[
    Decimal,
    PlainSerializer(serialize_decimal, return_type=str, when_used="json"),
]

Qty = Annotated[
    Decimal,
    PlainSerializer(lambda v: f"{v.normalize():f}", return_type=str, when_used="json"),
]


class Decision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    NEEDS_REVIEW = "needs_review"  # Sent to Human-in-the-Loop review queue
    REJECT = "reject"


class Severity(str, Enum):
    INFO = "info"
    REVIEW = "review"  # Requires Human-in-the-Loop reviewer attention
    REJECT = "reject"  # Hard violation; blocked automatically or flagged for rejection


class ReasonCode(str, Enum):
    # Missing fields / extraction issues (Human-in-the-Loop triggers)
    EXTRACTION_INCOMPLETE = "EXTRACTION_INCOMPLETE"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    NO_TEXT_LAYER = "NO_TEXT_LAYER"
    PO_MISSING = "PO_MISSING"
    PO_NOT_FOUND = "PO_NOT_FOUND"

    # Deterministic Arithmetic Mismatches
    LINE_AMOUNT_MISMATCH = "LINE_AMOUNT_MISMATCH"
    LINE_ROUNDING = "LINE_ROUNDING"
    SUBTOTAL_MISMATCH = "SUBTOTAL_MISMATCH"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"

    # 2-Way & 3-Way Reconciliation Discrepancies
    VENDOR_MISMATCH = "VENDOR_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    PRICE_VARIANCE = "PRICE_VARIANCE"
    PRICE_BELOW_PO = "PRICE_BELOW_PO"
    LINE_NOT_ON_PO = "LINE_NOT_ON_PO"
    QTY_EXCEEDS_ORDERED = "QTY_EXCEEDS_ORDERED"
    QTY_EXCEEDS_RECEIVED = "QTY_EXCEEDS_RECEIVED"
    PO_LINES_OPEN = "PO_LINES_OPEN"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"

    # Contract & Compliance Violations
    UNAUTHORIZED_SURCHARGE = "UNAUTHORIZED_SURCHARGE"
    CONTRACT_PRICE_EXCEEDED = "CONTRACT_PRICE_EXCEEDED"
    DISCOUNT_NOT_APPLIED = "DISCOUNT_NOT_APPLIED"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"
    PAYMENT_TERMS_MISMATCH = "PAYMENT_TERMS_MISMATCH"


class Reason(BaseModel):
    """Specific discrepancy or notice found during the audit."""
    model_config = ConfigDict(frozen=True)

    code: ReasonCode = Field(..., description="Machine-readable canonical discrepancy code")
    severity: Severity = Field(..., description="Actionable severity level: info, review, or reject")
    message: str = Field(..., description="Plain-English explanation of the finding")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary with exact diff values")
    requires_hitl: bool = Field(
        default=False,
        description="Explicit flag indicating this item requires Human-in-the-Loop intervention",
    )


class LineItem(BaseModel):
    """An individual invoice line item with strict decimal fields."""
    model_config = ConfigDict(frozen=True)

    line_number: int = Field(ge=1, description="1-indexed line item sequence number")
    description: str = Field(min_length=1, description="Product or service description")
    sku: Optional[str] = Field(default=None, description="Vendor part number or internal SKU identifier")
    quantity: Qty = Field(default=Decimal("1.0"), description="Quantity billed as decimal")
    unit_price: Money = Field(default=Decimal("0.00"), description="Unit rate charged as decimal")
    line_total: Money = Field(default=Decimal("0.00"), description="Total billed amount for this line")

    @field_validator("quantity", "unit_price", "line_total", mode="before")
    @classmethod
    def parse_to_decimal(cls, v: Any) -> Decimal:
        if v is None:
            return Decimal("0.00")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v).replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip())


class Invoice(BaseModel):
    """Strict model for an extracted invoice."""
    model_config = ConfigDict(extra="ignore")

    invoice_number: Optional[str] = Field(default=None, description="Unique invoice document identifier")
    vendor: Optional[str] = Field(default=None, description="Legal entity name of the supplier")
    vendor_address: Optional[str] = Field(default=None, description="Physical or billing street address of the supplier")
    date: Optional[str] = Field(default=None, description="Invoice issue date (YYYY-MM-DD)")
    po_number: Optional[str] = Field(default=None, description="Associated Purchase Order number reference")
    currency: str = Field(default="USD", description="3-letter ISO currency code")
    lines: List[LineItem] = Field(default_factory=list, description="List of itemized invoice line items")
    subtotal: Optional[Money] = Field(default=None, description="Printed invoice subtotal before tax and fees")
    tax: Optional[Money] = Field(default=Decimal("0.00"), description="Sales tax, VAT, or GST amount")
    discount: Optional[Money] = Field(default=Decimal("0.00"), description="Credit or early payment discount applied")
    freight_surcharges: Optional[Money] = Field(default=Decimal("0.00"), description="Shipping, fuel, or handling surcharges")
    total: Optional[Money] = Field(default=None, description="Final grand total amount due")

    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields for HITL review")

    @field_validator("subtotal", "tax", "discount", "freight_surcharges", "total", mode="before")
    @classmethod
    def parse_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        if isinstance(v, Decimal):
            return v
        val_str = str(v).replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
        if not val_str:
            return None
        return Decimal(val_str)


# --- Pydantic Structured Output Models for LLM Extraction & Analysis ---

class InvoiceLineExtractionSchema(BaseModel):
    """Structured line item extraction schema for GenAI extractors."""
    line_number: int = Field(default=1, description="Sequential line item number starting at 1")
    description: str = Field(..., description="Item or service description printed on the invoice")
    sku: Optional[str] = Field(default=None, description="Part number, catalog ID, or SKU code if present")
    quantity: str = Field(default="1.0", description="Numerical quantity purchased (e.g. '10.0')")
    unit_price: str = Field(..., description="Price charged per unit as decimal string without currency symbols")
    line_total: str = Field(..., description="Total line amount charged as decimal string")


class InvoiceExtractionSchema(BaseModel):
    """Structured document extraction schema for LLM invoice parsing."""
    invoice_number: Optional[str] = Field(default=None, description="Invoice reference number (e.g. 'INV-101')")
    vendor: Optional[str] = Field(default=None, description="Name of the selling company or supplier")
    vendor_address: Optional[str] = Field(default=None, description="Physical street or mailing address of supplier")
    po_number: Optional[str] = Field(default=None, description="Purchase order reference number if mentioned")
    date: Optional[str] = Field(default=None, description="Invoice date in YYYY-MM-DD or standard format")
    currency: str = Field(default="USD", description="3-letter currency code (e.g. 'USD', 'EUR', 'GBP')")
    lines: List[InvoiceLineExtractionSchema] = Field(default_factory=list, description="Array of extracted line items")
    subtotal: Optional[str] = Field(default=None, description="Subtotal amount before tax and fees as decimal string")
    tax: Optional[str] = Field(default="0.00", description="Sales tax, VAT, or GST amount as decimal string")
    discount: Optional[str] = Field(default="0.00", description="Discount or credit amount deducted as decimal string")
    freight_surcharges: Optional[str] = Field(default="0.00", description="Shipping, delivery, or fuel fee as decimal string")
    total: Optional[str] = Field(default=None, description="Grand total payable as decimal string")


class ContractClauseAnalysisSchema(BaseModel):
    """Structured legal compliance analysis output from LLM."""
    is_violation: bool = Field(..., description="True if the invoice charge or term breaches the contract clause")
    severity: str = Field(default="info", description="Classification of severity: 'reject', 'review', or 'info'")
    explanation: str = Field(..., description="Clear legal explanation citing specific terms in the clause")


# --- Relational ERP Table Projection Models ---

class PurchaseOrderTableRow(BaseModel):
    """Relational table projection for ERP Purchase Order views."""
    po_number: str = Field(..., description="Master Purchase Order number")
    vendor: str = Field(..., description="Supplier vendor name")
    currency: str = Field(..., description="Currency code")
    receipt_required: bool = Field(..., description="Whether warehouse 3-way goods receipt is mandatory")
    line_number: int = Field(..., description="PO item line sequence number")
    sku: str = Field(..., description="Stock keeping unit or catalog part number")
    description: str = Field(..., description="Authorized item description")
    quantity: str = Field(..., description="Authorized purchase quantity")
    unit_price: str = Field(..., description="Contracted price per unit")
    line_total: str = Field(..., description="Calculated total line value (quantity x price)")


class GoodsReceiptTableRow(BaseModel):
    """Relational table projection for Warehouse Goods Receipt views."""
    receipt_id: str = Field(..., description="Warehouse receipt document identifier")
    po_number: str = Field(..., description="Referenced Purchase Order number")
    receipt_date: str = Field(..., description="Date shipment was verified in warehouse")
    line_number: int = Field(..., description="Corresponding PO line item number")
    sku: str = Field(..., description="Received item SKU identifier")
    quantity_received: str = Field(..., description="Physical quantity verified and accepted")
    status: str = Field(default="Verified in Warehouse", description="Current warehouse receiving status")


class POLine(BaseModel):
    """A line item on an authorized Purchase Order."""
    model_config = ConfigDict(frozen=True)

    line_number: int = Field(ge=1, description="Line number")
    sku: Optional[str] = Field(default=None, description="SKU code")
    description: str = Field(..., description="Item description")
    quantity: Qty = Field(..., description="Ordered quantity")
    unit_price: Money = Field(..., description="Authorized unit price")
    tax_rate: Decimal = Field(default=Decimal("0.00"), description="Applicable tax rate")

    @field_validator("quantity", "unit_price", mode="before")
    @classmethod
    def parse_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v).replace(",", "").replace("$", "").strip())


class PurchaseOrder(BaseModel):
    """Master authorized Purchase Order document."""
    model_config = ConfigDict(extra="ignore")

    po_number: str = Field(..., description="PO number")
    vendor: str = Field(..., description="Authorized vendor name")
    vendor_address: Optional[str] = Field(default=None, description="Authorized vendor physical or billing street address")
    date: Optional[str] = Field(default=None, description="Purchase order issuance date (YYYY-MM-DD)")
    currency: str = Field(default="USD", description="Currency")
    receipt_required: bool = Field(default=True, description="Whether 3-way match is required")
    lines: List[POLine] = Field(default_factory=list, description="Authorized PO line items")
    max_tolerated_price_variance: Decimal = Field(default=Decimal("0.00"), description="Tolerated price drift")

    def to_table_rows(self) -> List[Dict[str, Any]]:
        """Project PO lines into flat relational table rows."""
        rows = []
        for line in self.lines:
            row = PurchaseOrderTableRow(
                po_number=self.po_number,
                vendor=self.vendor,
                currency=self.currency,
                receipt_required=self.receipt_required,
                line_number=line.line_number,
                sku=line.sku or "N/A",
                description=line.description,
                quantity=str(line.quantity),
                unit_price=str(line.unit_price),
                line_total=str((line.quantity * line.unit_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            )
            rows.append(row.model_dump())
        return rows


class ReceiptLine(BaseModel):
    """Line item recording warehouse goods receipt."""
    model_config = ConfigDict(frozen=True)

    line_number: int = Field(..., description="PO line number")
    sku: Optional[str] = Field(default=None, description="SKU code")
    quantity_received: Qty = Field(..., description="Verified received quantity")

    @field_validator("quantity_received", mode="before")
    @classmethod
    def parse_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v).replace(",", "").replace("$", "").strip())


class GoodsReceipt(BaseModel):
    """Proof of physical warehouse receipt for 3-way matching."""
    model_config = ConfigDict(extra="ignore")

    receipt_id: str = Field(..., description="Receipt ID")
    po_number: str = Field(..., description="PO number reference")
    receipt_date: str = Field(..., description="Date of physical delivery")
    lines: List[ReceiptLine] = Field(default_factory=list, description="Received items")

    def to_table_rows(self) -> List[Dict[str, Any]]:
        """Project Goods Receipt lines into flat relational table rows."""
        rows = []
        for line in self.lines:
            row = GoodsReceiptTableRow(
                receipt_id=self.receipt_id,
                po_number=self.po_number,
                receipt_date=self.receipt_date,
                line_number=line.line_number,
                sku=line.sku or "N/A",
                quantity_received=str(line.quantity_received),
                status="Verified in Warehouse",
            )
            rows.append(row.model_dump())
        return rows


class ContractClause(BaseModel):
    """Contractual term, pricing rule, or SLA condition."""
    clause_id: str = Field(..., description="Unique legal clause identifier")
    vendor: str = Field(..., description="Contracting vendor name")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Legal clause text")
    category: str = Field(default="pricing", description="Clause category: pricing, fees, terms, penalties")
    prohibits_surcharges: bool = Field(default=False, description="Whether separate freight/fuel fees are banned")
    allowed_payment_terms: Optional[str] = Field(default=None, description="Contractual payment terms")
    discount_percentage: Optional[Decimal] = Field(default=None, description="Agreed volume/early payment discount %")


class EvidenceItem(BaseModel):
    """Concrete evidence attaching document proof to an audit finding."""
    category: str = Field(..., description="Evidence category")
    source_document: str = Field(..., description="Source document name")
    reference_id: str = Field(..., description="Document or clause reference ID")
    citation_text: str = Field(..., description="Textual excerpt or calculation citation")
    confidence: float = Field(default=1.0, description="Attribution confidence score")


class ReviewAction(str, Enum):
    APPROVE = "approve"
    FLAG = "flag_for_correction"
    REJECT = "reject"


class ReviewDecision(BaseModel):
    """Human-in-the-Loop decision recorded by the auditor."""
    reviewer_id: str = Field(..., description="Identifier of the human auditor")
    action: ReviewAction = Field(..., description="Auditor determination")
    notes: str = Field(..., description="Auditor justification or dispute notes")
    overridden_reasons: List[str] = Field(default_factory=list, description="Reason codes overridden by the reviewer")
    corrected_data: Optional[Dict[str, Any]] = Field(default=None, description="Auditor keyed corrections")
    timestamp: str = Field(..., description="ISO 8601 audit timestamp")


class AuditResult(BaseModel):
    """Comprehensive result of the full audit pipeline."""
    invoice_id: str = Field(..., description="Audited invoice identifier")
    decision: Decision = Field(..., description="Recommended audit outcome")
    reasons: List[Reason] = Field(default_factory=list, description="List of detected discrepancy reasons")
    evidence_bundle: List[EvidenceItem] = Field(default_factory=list, description="Attached evidence citations")
    risk_score: float = Field(ge=0.0, le=100.0, default=0.0, description="Computed risk exposure score (0-100)")
    requires_human_review: bool = Field(default=False, description="Whether human review queue routing is required")
    human_in_loop_message: Optional[str] = Field(default=None, description="Clear guidance alert for the human reviewer")
    financial_exposure: Money = Field(default=Decimal("0.00"), description="Net financial variance/overcharge detected")
    investigation_summary: str = Field(default="", description="Multi-agent executive investigation summary")
    human_decision: Optional[ReviewDecision] = Field(default=None, description="Recorded human review action if completed")
    matched_lines_map: Dict[int, int] = Field(default_factory=dict, description="Map of invoice line number to PO line number")
