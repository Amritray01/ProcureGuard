"""Deterministic arithmetic and integrity validator for invoices.

Performs exact decimal mathematical checks without any LLM approximations.
Detects missing mandatory fields and explicitly attaches Human-in-the-Loop review alerts.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple
from src.core.schema import Invoice, Reason, ReasonCode, Severity

CENT = Decimal("0.01")
ROUNDING_TOLERANCE = Decimal("0.02")


def quantize_money(amount: Decimal) -> Decimal:
    """Round to 2 decimal places using standard financial round half up."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def check_missing_fields(invoice: Invoice) -> List[Reason]:
    """Check for missing or incomplete required fields.
    
    If any critical field is missing, emits an explicit Human-in-the-Loop message
    and flags severity as REVIEW.
    """
    reasons: List[Reason] = []
    missing = list(invoice.missing_fields)

    if not invoice.invoice_number:
        missing.append("invoice_number")
    if not invoice.vendor:
        missing.append("vendor")
    if not invoice.lines:
        missing.append("line_items")
    if invoice.total is None:
        missing.append("total_amount")

    # Remove duplicates preserving order
    dedup_missing = list(dict.fromkeys(missing))

    for field in dedup_missing:
        reasons.append(
            Reason(
                code=ReasonCode.MISSING_REQUIRED_FIELD,
                severity=Severity.REVIEW,
                message=(
                    f"Missing required field '{field}'. Please use Human-In-The-Loop review "
                    f"to manually verify or key in this field from the document."
                ),
                details={"field": field, "hitl_required": True},
                requires_hitl=True,
            )
        )

    if not invoice.po_number:
        reasons.append(
            Reason(
                code=ReasonCode.PO_MISSING,
                severity=Severity.REVIEW,
                message=(
                    "No Purchase Order (PO) number was found on the invoice. "
                    "Use Human-In-The-Loop review to match this invoice against open vendor POs."
                ),
                details={"hitl_required": True},
                requires_hitl=True,
            )
        )

    return reasons


def validate_arithmetic(invoice: Invoice) -> Tuple[List[Reason], Decimal]:
    """Perform deterministic mathematical audits on line items, subtotals, and total.
    
    Returns:
        Tuple of (list of reasons/discrepancies, total net discrepancy amount).
    """
    reasons: List[Reason] = []
    total_discrepancy = Decimal("0.00")

    # 1. Line item math: quantity x unit_price vs line_total
    computed_subtotal = Decimal("0.00")

    for line in invoice.lines:
        expected_line_amount = quantize_money(line.quantity * line.unit_price)
        diff = abs(expected_line_amount - line.line_total)

        if diff > ROUNDING_TOLERANCE:
            reasons.append(
                Reason(
                    code=ReasonCode.LINE_AMOUNT_MISMATCH,
                    severity=Severity.REVIEW,
                    message=(
                        f"Line #{line.line_number} ('{line.description}'): Printed line total "
                        f"{line.line_total} does not match Quantity ({line.quantity}) x "
                        f"Unit Price ({line.unit_price}) = {expected_line_amount}. "
                        f"Difference: {diff}. Requires Human-In-The-Loop review."
                    ),
                    details={
                        "line_number": line.line_number,
                        "printed_total": str(line.line_total),
                        "calculated_total": str(expected_line_amount),
                        "difference": str(diff),
                    },
                    requires_hitl=True,
                )
            )
            total_discrepancy += diff
        elif diff > Decimal("0.00"):
            reasons.append(
                Reason(
                    code=ReasonCode.LINE_ROUNDING,
                    severity=Severity.INFO,
                    message=f"Line #{line.line_number} has minor rounding difference of {diff}.",
                    details={"line_number": line.line_number, "diff": str(diff)},
                    requires_hitl=False,
                )
            )

        computed_subtotal += line.line_total

    computed_subtotal = quantize_money(computed_subtotal)

    # 2. Subtotal check
    if invoice.subtotal is not None:
        subtotal_diff = abs(computed_subtotal - invoice.subtotal)
        if subtotal_diff > ROUNDING_TOLERANCE:
            reasons.append(
                Reason(
                    code=ReasonCode.SUBTOTAL_MISMATCH,
                    severity=Severity.REVIEW,
                    message=(
                        f"Sum of line totals ({computed_subtotal}) does not equal printed "
                        f"subtotal ({invoice.subtotal}). Discrepancy: {subtotal_diff}. "
                        f"Requires Human-In-The-Loop review."
                    ),
                    details={
                        "computed_subtotal": str(computed_subtotal),
                        "printed_subtotal": str(invoice.subtotal),
                        "difference": str(subtotal_diff),
                    },
                    requires_hitl=True,
                )
            )
            total_discrepancy += subtotal_diff

    # 3. Grand total check: Subtotal - Discount + Tax + Freight = Total
    effective_subtotal = invoice.subtotal if invoice.subtotal is not None else computed_subtotal
    tax = invoice.tax if invoice.tax is not None else Decimal("0.00")
    discount = invoice.discount if invoice.discount is not None else Decimal("0.00")
    freight = invoice.freight_surcharges if invoice.freight_surcharges is not None else Decimal("0.00")

    expected_grand_total = quantize_money(effective_subtotal - discount + tax + freight)

    if invoice.total is not None:
        total_diff = abs(expected_grand_total - invoice.total)
        if total_diff > ROUNDING_TOLERANCE:
            reasons.append(
                Reason(
                    code=ReasonCode.TOTAL_MISMATCH,
                    severity=Severity.REVIEW,
                    message=(
                        f"Arithmetic mismatch on Grand Total: Printed total is {invoice.total}, but "
                        f"Subtotal ({effective_subtotal}) - Discount ({discount}) + Tax ({tax}) + "
                        f"Freight/Surcharges ({freight}) = {expected_grand_total}. "
                        f"Variance of {total_diff}. Requires Human-In-The-Loop review."
                    ),
                    details={
                        "printed_total": str(invoice.total),
                        "expected_total": str(expected_grand_total),
                        "variance": str(total_diff),
                    },
                    requires_hitl=True,
                )
            )
            total_discrepancy += total_diff

    return reasons, quantize_money(total_discrepancy)
