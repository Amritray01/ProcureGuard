"""Strict data models and schemas for the Procurement Invoice & Contract Auditor.

Money and quantities are strictly represented using decimal.Decimal to eliminate
floating-point rounding errors and ensure 100% deterministic auditing.
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

    code: ReasonCode
    severity: Severity
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    requires_hitl: bool = Field(
        default=False,
        description="Explicit flag indicating this item requires Human-in-the-Loop intervention",
    )


class LineItem(BaseModel):
    """An individual invoice line item with strict decimal fields."""
    model_config = ConfigDict(frozen=True)

    line_number: int = Field(ge=1)
    description: str = Field(min_length=1)
    sku: Optional[str] = None
    quantity: Qty = Field(default=Decimal("1.0"))
    unit_price: Money = Field(default=Decimal("0.00"))
    line_total: Money = Field(default=Decimal("0.00"))

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

    invoice_number: Optional[str] = None
    vendor: Optional[str] = None
    date: Optional[str] = None
    po_number: Optional[str] = None
    currency: str = Field(default="USD")
    lines: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[Money] = None
    tax: Optional[Money] = Decimal("0.00")
    discount: Optional[Money] = Decimal("0.00")
    freight_surcharges: Optional[Money] = Decimal("0.00")
    total: Optional[Money] = None

    # Track missing fields explicitly for Human-In-The-Loop review
    missing_fields: List[str] = Field(default_factory=list)

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


class POLine(BaseModel):
    """A line item on an authorized Purchase Order."""
    model_config = ConfigDict(frozen=True)

    line_number: int = Field(ge=1)
    sku: Optional[str] = None
    description: str
    quantity: Qty
    unit_price: Money
    tax_rate: Decimal = Decimal("0.00")

    @field_validator("quantity", "unit_price", mode="before")
    @classmethod
    def parse_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v).replace(",", "").replace("$", "").strip())


class PurchaseOrder(BaseModel):
    """Master authorized Purchase Order document."""
    model_config = ConfigDict(extra="ignore")

    po_number: str
    vendor: str
    currency: str = "USD"
    receipt_required: bool = True
    lines: List[POLine] = Field(default_factory=list)
    max_tolerated_price_variance: Decimal = Decimal("0.00")


class ReceiptLine(BaseModel):
    """Line item recording warehouse goods receipt."""
    model_config = ConfigDict(frozen=True)

    line_number: int
    sku: Optional[str] = None
    quantity_received: Qty

    @field_validator("quantity_received", mode="before")
    @classmethod
    def parse_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v).replace(",", "").replace("$", "").strip())


class GoodsReceipt(BaseModel):
    """Proof of physical warehouse receipt for 3-way matching."""
    model_config = ConfigDict(extra="ignore")

    receipt_id: str
    po_number: str
    receipt_date: str
    lines: List[ReceiptLine] = Field(default_factory=list)


class ContractClause(BaseModel):
    """Contractual term, pricing rule, or SLA condition."""
    clause_id: str
    vendor: str
    title: str
    content: str
    category: str = "pricing"  # pricing, fees, terms, penalties
    prohibits_surcharges: bool = False
    allowed_payment_terms: Optional[str] = None
    discount_percentage: Optional[Decimal] = None


class EvidenceItem(BaseModel):
    """Concrete evidence attaching document proof to an audit finding."""
    category: str
    source_document: str  # Invoice, PO, Goods Receipt, or Contract
    reference_id: str
    citation_text: str
    confidence: float = 1.0


class ReviewAction(str, Enum):
    APPROVE = "approve"
    FLAG = "flag_for_correction"
    REJECT = "reject"


class ReviewDecision(BaseModel):
    """Human-in-the-Loop decision recorded by the auditor."""
    reviewer_id: str
    action: ReviewAction
    notes: str
    overridden_reasons: List[str] = Field(default_factory=list)
    corrected_data: Optional[Dict[str, Any]] = None
    timestamp: str


class AuditResult(BaseModel):
    """Comprehensive result of the full audit pipeline."""
    invoice_id: str
    decision: Decision
    reasons: List[Reason]
    evidence_bundle: List[EvidenceItem]
    risk_score: float = Field(ge=0.0, le=100.0, default=0.0)
    requires_human_review: bool = False
    human_in_loop_message: Optional[str] = None
    financial_exposure: Money = Decimal("0.00")
    investigation_summary: str = ""
    human_decision: Optional[ReviewDecision] = None
