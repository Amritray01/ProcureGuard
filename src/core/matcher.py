"""Deterministic 2-Way and 3-Way Matching Engine with Stateful Cumulative Ledger.

Compares invoice line items strictly against authorized Purchase Orders (PO) and
warehouse Goods Receipts. Remembers historical billings across multiple invoices to
catch duplicate submissions, cumulative overbilling, and unreceived shipments.
"""

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from src.core.schema import (
    GoodsReceipt,
    Invoice,
    POLine,
    PurchaseOrder,
    Reason,
    ReasonCode,
    Severity,
)
from src.core.arithmetic import quantize_money


class CumulativeLedger:
    """Stateful ledger that tracks historical billings per PO line and detects duplicates."""

    def __init__(self) -> None:
        # Seen invoices key: (normalized_vendor, normalized_invoice_num)
        self.seen_invoices: Set[Tuple[str, str]] = set()

        # Near-duplicate lookup: (normalized_vendor, currency, total) -> invoice_num
        self.amount_fingerprints: Dict[Tuple[str, str, str], str] = {}

        # Cumulative quantity billed: (po_number, po_line_number) -> Decimal
        self.billed_quantities: Dict[Tuple[str, int], Decimal] = defaultdict(lambda: Decimal("0.0"))

    def check_duplicate(self, invoice: Invoice) -> Optional[Reason]:
        """Check for exact duplicate and near duplicate invoice submissions."""
        if not invoice.invoice_number or not invoice.vendor:
            return None

        vendor_key = invoice.vendor.strip().lower()
        inv_num_key = invoice.invoice_number.strip().lower()

        # 1. Exact Duplicate
        if (vendor_key, inv_num_key) in self.seen_invoices:
            return Reason(
                code=ReasonCode.DUPLICATE_INVOICE,
                severity=Severity.REJECT,
                message=(
                    f"Duplicate invoice detected: Vendor '{invoice.vendor}' and Invoice "
                    f"Number '{invoice.invoice_number}' were already processed previously. "
                    f"Immediate rejection recommended; verify with Human-In-The-Loop review."
                ),
                details={"vendor": invoice.vendor, "invoice_number": invoice.invoice_number},
                requires_hitl=True,
            )

        # 2. Possible duplicate (different invoice number, same vendor, currency, and exact total)
        if invoice.total is not None:
            fp_key = (vendor_key, invoice.currency.upper(), str(invoice.total))
            if fp_key in self.amount_fingerprints:
                prior_inv = self.amount_fingerprints[fp_key]
                return Reason(
                    code=ReasonCode.POSSIBLE_DUPLICATE,
                    severity=Severity.REVIEW,
                    message=(
                        f"Possible duplicate invoice: Vendor '{invoice.vendor}', currency "
                        f"{invoice.currency}, and total {invoice.total} match previously "
                        f"processed invoice '{prior_inv}'. Requires Human-In-The-Loop review."
                    ),
                    details={"prior_invoice": prior_inv, "total": str(invoice.total)},
                    requires_hitl=True,
                )

        return None

    def record_invoice(self, invoice: Invoice) -> None:
        """Record successful or processed invoice into the ledger."""
        if invoice.invoice_number and invoice.vendor:
            vendor_key = invoice.vendor.strip().lower()
            inv_num_key = invoice.invoice_number.strip().lower()
            self.seen_invoices.add((vendor_key, inv_num_key))

            if invoice.total is not None:
                fp_key = (vendor_key, invoice.currency.upper(), str(invoice.total))
                self.amount_fingerprints[fp_key] = invoice.invoice_number

    def record_po_billing(self, po_number: str, line_no: int, quantity: Decimal) -> None:
        """Add to the cumulative billed quantity for a PO line."""
        self.billed_quantities[(po_number.upper(), line_no)] += quantity

    def get_billed_quantity(self, po_number: str, line_no: int) -> Decimal:
        """Get the cumulative quantity already billed for a PO line."""
        return self.billed_quantities[(po_number.upper(), line_no)]


class DeterministicMatcher:
    """Executes 2-way and 3-way reconciliation against POs and Goods Receipts."""

    def __init__(self, ledger: Optional[CumulativeLedger] = None) -> None:
        self.ledger = ledger or CumulativeLedger()

    def match(
        self,
        invoice: Invoice,
        po: Optional[PurchaseOrder] = None,
        receipt: Optional[GoodsReceipt] = None,
    ) -> Tuple[List[Reason], Decimal]:
        """Perform deterministic matching.
        
        Returns:
            Tuple of (reasons, total financial variance detected)
        """
        reasons: List[Reason] = []
        financial_variance = Decimal("0.00")

        # 1. Duplicate Check via Stateful Ledger
        dup_reason = self.ledger.check_duplicate(invoice)
        if dup_reason:
            reasons.append(dup_reason)
            if dup_reason.severity == Severity.REJECT:
                # Fatal duplicate; reject immediately
                return reasons, invoice.total or Decimal("0.00")

        # 2. Check PO Availability
        if not po:
            reasons.append(
                Reason(
                    code=ReasonCode.PO_NOT_FOUND,
                    severity=Severity.REVIEW,
                    message=(
                        f"Purchase Order '{invoice.po_number or 'UNKNOWN'}' was not found in the "
                        f"system. Use Human-In-The-Loop review to link an approved PO."
                    ),
                    details={"po_number": invoice.po_number, "hitl_required": True},
                    requires_hitl=True,
                )
            )
            return reasons, invoice.total or Decimal("0.00")

        # 3. Vendor Matching
        if invoice.vendor and po.vendor:
            norm_inv_v = invoice.vendor.strip().lower()
            norm_po_v = po.vendor.strip().lower()
            if norm_inv_v != norm_po_v and norm_inv_v not in norm_po_v and norm_po_v not in norm_inv_v:
                reasons.append(
                    Reason(
                        code=ReasonCode.VENDOR_MISMATCH,
                        severity=Severity.REJECT,
                        message=(
                            f"Vendor mismatch: Invoice vendor '{invoice.vendor}' does not match "
                            f"PO vendor '{po.vendor}'. Requires Human-In-The-Loop review."
                        ),
                        details={"invoice_vendor": invoice.vendor, "po_vendor": po.vendor},
                        requires_hitl=True,
                    )
                )

        # 4. Currency Matching
        if invoice.currency and po.currency and invoice.currency.upper() != po.currency.upper():
            reasons.append(
                Reason(
                    code=ReasonCode.CURRENCY_MISMATCH,
                    severity=Severity.REVIEW,
                    message=(
                        f"Currency mismatch: Invoice currency '{invoice.currency}' differs from "
                        f"PO currency '{po.currency}'. Human-In-The-Loop review required."
                    ),
                    details={"invoice_currency": invoice.currency, "po_currency": po.currency},
                    requires_hitl=True,
                )
            )

        # 5. Map PO Lines for matching (by SKU if present, else by line_number or description)
        po_lines_by_sku: Dict[str, POLine] = {line.sku.upper(): line for line in po.lines if line.sku}
        po_lines_by_num: Dict[int, POLine] = {line.line_number: line for line in po.lines}
        matched_po_line_nums: Set[int] = set()

        # Map Goods Receipts by line number
        receipt_qty_by_line: Dict[int, Decimal] = defaultdict(Decimal)
        if receipt:
            for r_line in receipt.lines:
                receipt_qty_by_line[r_line.line_number] += r_line.quantity_received

        # 6. Line Item Reconciliation
        for inv_line in invoice.lines:
            matched_po_line: Optional[POLine] = None

            # Try SKU match first
            if inv_line.sku and inv_line.sku.upper() in po_lines_by_sku:
                matched_po_line = po_lines_by_sku[inv_line.sku.upper()]
            # Try line number match
            elif inv_line.line_number in po_lines_by_num:
                matched_po_line = po_lines_by_num[inv_line.line_number]
            else:
                # Try fuzzy description fallback
                for p_line in po.lines:
                    if p_line.line_number not in matched_po_line_nums:
                        if (
                            inv_line.description.lower() in p_line.description.lower()
                            or p_line.description.lower() in inv_line.description.lower()
                        ):
                            matched_po_line = p_line
                            break

            if not matched_po_line:
                reasons.append(
                    Reason(
                        code=ReasonCode.LINE_NOT_ON_PO,
                        severity=Severity.REVIEW,
                        message=(
                            f"Invoice Line #{inv_line.line_number} ('{inv_line.description}') "
                            f"has no corresponding line on PO #{po.po_number}. "
                            f"Requires Human-In-The-Loop review."
                        ),
                        details={"line_number": inv_line.line_number, "description": inv_line.description},
                        requires_hitl=True,
                    )
                )
                financial_variance += inv_line.line_total
                continue

            matched_po_line_nums.add(matched_po_line.line_number)

            # A. Price Variance Check
            price_variance = inv_line.unit_price - matched_po_line.unit_price
            if price_variance > po.max_tolerated_price_variance:
                diff_amount = quantize_money(price_variance * inv_line.quantity)
                reasons.append(
                    Reason(
                        code=ReasonCode.PRICE_VARIANCE,
                        severity=Severity.REVIEW,
                        message=(
                            f"Line #{inv_line.line_number} unit price ({inv_line.unit_price}) exceeds "
                            f"authorized PO rate ({matched_po_line.unit_price}) by {price_variance}. "
                            f"Overcharge impact: {diff_amount}. Requires Human-In-The-Loop review."
                        ),
                        details={
                            "line_number": inv_line.line_number,
                            "invoice_unit_price": str(inv_line.unit_price),
                            "po_unit_price": str(matched_po_line.unit_price),
                            "overcharge": str(diff_amount),
                        },
                        requires_hitl=True,
                    )
                )
                financial_variance += diff_amount
            elif price_variance < Decimal("0.00"):
                reasons.append(
                    Reason(
                        code=ReasonCode.PRICE_BELOW_PO,
                        severity=Severity.INFO,
                        message=(
                            f"Line #{inv_line.line_number} unit price ({inv_line.unit_price}) is "
                            f"below authorized PO rate ({matched_po_line.unit_price}). Favorable discount."
                        ),
                        details={"line_number": inv_line.line_number, "saving": str(abs(price_variance))},
                        requires_hitl=False,
                    )
                )

            # B. Cumulative Quantity Check (2-Way)
            prior_billed = self.ledger.get_billed_quantity(po.po_number, matched_po_line.line_number)
            new_cumulative_qty = prior_billed + inv_line.quantity

            if new_cumulative_qty > matched_po_line.quantity:
                overbilled_qty = new_cumulative_qty - matched_po_line.quantity
                overbilled_amount = quantize_money(overbilled_qty * inv_line.unit_price)
                reasons.append(
                    Reason(
                        code=ReasonCode.QTY_EXCEEDS_ORDERED,
                        severity=Severity.REJECT,
                        message=(
                            f"Quantity overbilling on Line #{inv_line.line_number}: Billed quantity "
                            f"{inv_line.quantity} + prior billings ({prior_billed}) = {new_cumulative_qty}, "
                            f"which exceeds ordered quantity ({matched_po_line.quantity}) by {overbilled_qty}. "
                            f"Requires Human-In-The-Loop review."
                        ),
                        details={
                            "po_line": matched_po_line.line_number,
                            "ordered_qty": str(matched_po_line.quantity),
                            "prior_billed_qty": str(prior_billed),
                            "current_billed_qty": str(inv_line.quantity),
                            "overbilled_qty": str(overbilled_qty),
                            "overbilled_amount": str(overbilled_amount),
                        },
                        requires_hitl=True,
                    )
                )
                financial_variance += overbilled_amount

            # C. Goods Receipt Check (3-Way)
            if po.receipt_required:
                if not receipt:
                    reasons.append(
                        Reason(
                            code=ReasonCode.QTY_EXCEEDS_RECEIVED,
                            severity=Severity.REVIEW,
                            message=(
                                f"3-Way match alert on Line #{inv_line.line_number}: PO requires goods "
                                f"receipt, but no Goods Receipt found. Requires Human-In-The-Loop review."
                            ),
                            details={"line_number": inv_line.line_number},
                            requires_hitl=True,
                        )
                    )
                else:
                    received_qty = receipt_qty_by_line.get(matched_po_line.line_number, Decimal("0.0"))
                    if new_cumulative_qty > received_qty:
                        unreceived_qty = new_cumulative_qty - received_qty
                        reasons.append(
                            Reason(
                                code=ReasonCode.QTY_EXCEEDS_RECEIVED,
                                severity=Severity.REVIEW,
                                message=(
                                    f"Line #{inv_line.line_number}: Billed quantity ({new_cumulative_qty}) "
                                    f"exceeds warehouse verified received quantity ({received_qty}) by "
                                    f"{unreceived_qty}. Goods not yet received. Requires Human-In-The-Loop review."
                                ),
                                details={
                                    "line_number": inv_line.line_number,
                                    "received_qty": str(received_qty),
                                    "unreceived_qty": str(unreceived_qty),
                                },
                                requires_hitl=True,
                            )
                        )

        # 7. Check for unbilled PO lines (partial billing)
        unbilled_lines = [p.line_number for p in po.lines if p.line_number not in matched_po_line_nums]
        if unbilled_lines:
            reasons.append(
                Reason(
                    code=ReasonCode.PO_LINES_OPEN,
                    severity=Severity.INFO,
                    message=f"PO #{po.po_number} has open unbilled lines: {unbilled_lines}.",
                    details={"open_po_lines": unbilled_lines},
                    requires_hitl=False,
                )
            )

        return reasons, quantize_money(financial_variance)
