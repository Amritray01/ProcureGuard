"""Dynamic PDF Extractor combining PyMuPDF visual line reconstruction with LLM understanding.

Handles custom, arbitrary PDF invoice layouts.
If LLM API is available, uses Pydantic structured extraction.
If offline, uses visual horizontal coordinate line parser.
Explicitly detects scanned PDFs (no text layer) and missing required fields,
triggering Human-in-the-Loop review.
"""

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pymupdf

from src.core.schema import Invoice, LineItem, Reason, ReasonCode, Severity
from src.core.llm import llm_client


class PDFExtractor:
    """Extracts structured invoice data dynamically from any PDF format."""

    def extract_from_path(self, pdf_path: Union[str, Path]) -> Tuple[Invoice, List[Reason]]:
        """Extract invoice data from PDF file path."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        doc = pymupdf.open(str(path))
        return self._extract_from_doc(doc)

    def extract_from_bytes(self, pdf_bytes: bytes) -> Tuple[Invoice, List[Reason]]:
        """Extract invoice data from in-memory PDF bytes."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return self._extract_from_doc(doc)

    def _extract_from_doc(self, doc: pymupdf.Document) -> Tuple[Invoice, List[Reason]]:
        reasons: List[Reason] = []
        total_pages = len(doc)
        all_lines: List[str] = []
        raw_full_text = ""

        for page in doc:
            raw_full_text += page.get_text() + "\n"
            words = page.get_text("words")
            if words:
                # Group words on the same horizontal line by rounding Y0 coordinate
                lines_by_y: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
                for w in words:
                    y_approx = round(w[1] / 3) * 3
                    lines_by_y[y_approx].append((w[0], w[4]))

                for y in sorted(lines_by_y.keys()):
                    sorted_words = [t[1] for t in sorted(lines_by_y[y], key=lambda x: x[0])]
                    all_lines.append(" ".join(sorted_words))

        doc.close()

        # Check for scanned PDF / missing text layer
        if not all_lines or len(raw_full_text.strip()) < 15:
            reasons.append(
                Reason(
                    code=ReasonCode.NO_TEXT_LAYER,
                    severity=Severity.REVIEW,
                    message=(
                        "The uploaded PDF has no extractable text layer (likely a scanned image). "
                        "Please use Human-In-The-Loop review to manually key in the invoice values."
                    ),
                    details={"pages": total_pages, "hitl_required": True},
                    requires_hitl=True,
                )
            )
            invoice = Invoice(
                invoice_number=None,
                vendor=None,
                lines=[],
                missing_fields=["invoice_number", "vendor", "line_items", "total_amount"],
            )
            return invoice, reasons

        reconstructed_text = "\n".join(all_lines)

        # 1. Try Dynamic Extraction via LLM Structured Output if configured
        if llm_client.has_api_key():
            extracted = llm_client.extract_invoice_dynamic(reconstructed_text)
            if extracted and extracted.get("lines"):
                return self._build_invoice_from_dict(extracted), reasons

        # 2. Robust visual line parsing fallback
        invoice = self._parse_lines(all_lines, reconstructed_text)
        return invoice, reasons

    def _build_invoice_from_dict(self, extracted: Dict[str, Any]) -> Invoice:
        line_items: List[LineItem] = []
        for raw_line in extracted.get("lines", []):
            try:
                line_items.append(
                    LineItem(
                        line_number=int(raw_line.get("line_number", 1)),
                        description=str(raw_line.get("description", "Item")).strip(),
                        sku=raw_line.get("sku"),
                        quantity=Decimal(str(raw_line.get("quantity", "1.0"))),
                        unit_price=Decimal(str(raw_line.get("unit_price", "0.00")).replace("$", "")),
                        line_total=Decimal(str(raw_line.get("line_total", "0.00")).replace("$", "")),
                    )
                )
            except Exception:
                continue

        missing: List[str] = []
        inv_num = extracted.get("invoice_number")
        if not inv_num:
            missing.append("invoice_number")

        vendor = extracted.get("vendor")
        if not vendor:
            missing.append("vendor")

        po_num = extracted.get("po_number")
        if not po_num:
            missing.append("po_number")

        if not line_items:
            missing.append("line_items")

        total_val = None
        if extracted.get("total") is not None:
            try:
                total_val = Decimal(str(extracted["total"]).replace("$", "").replace(",", ""))
            except Exception:
                total_val = None

        if total_val is None:
            missing.append("total_amount")

        def parse_dec(val: Any) -> Optional[Decimal]:
            if val is None:
                return None
            try:
                return Decimal(str(val).replace("$", "").replace(",", "").strip())
            except Exception:
                return None

        return Invoice(
            invoice_number=inv_num,
            vendor=vendor,
            vendor_address=extracted.get("vendor_address"),
            date=extracted.get("date"),
            po_number=po_num,
            currency=extracted.get("currency", "USD"),
            lines=line_items,
            subtotal=parse_dec(extracted.get("subtotal")),
            tax=parse_dec(extracted.get("tax")) or Decimal("0.00"),
            discount=parse_dec(extracted.get("discount")) or Decimal("0.00"),
            freight_surcharges=parse_dec(extracted.get("freight_surcharges")) or Decimal("0.00"),
            total=total_val,
            missing_fields=missing,
        )

    def _parse_lines(self, lines: List[str], full_text: str) -> Invoice:
        missing: List[str] = []

        # 1. Invoice Number
        inv_match = re.search(r"(?:invoice\s*(?:#|no\.?|num(?:ber)?)|inv#)\s*[:\s]*([A-Za-z0-9\-_]+)", full_text, re.IGNORECASE)
        inv_number = inv_match.group(1).strip() if inv_match else None
        if not inv_number:
            missing.append("invoice_number")

        # 2. PO Number
        po_match = re.search(r"(?:p\.?o\.?\s*(?:#|no\.?|num(?:ber)?)|purchase\s*order\s*#?)\s*[:\s]*([A-Za-z0-9\-_]+)", full_text, re.IGNORECASE)
        po_number = po_match.group(1).strip() if po_match else None
        if not po_number:
            missing.append("po_number")

        # 3. Vendor
        vendor_match = re.search(r"(?:vendor|supplier|from|billed\s*by)\s*[:\s]*([A-Za-z0-9\s,\.&]+?)(?:\s+invoice|\s+date|\n|$)", full_text, re.IGNORECASE)
        vendor = vendor_match.group(1).strip() if vendor_match else None
        if not vendor:
            for l in lines[:3]:
                if not re.search(r"(commercial|invoice|date|order|tax)", l, re.IGNORECASE) and len(l) > 3:
                    vendor = l.strip()
                    break
        if not vendor:
            missing.append("vendor")

        # 4. Currency
        currency = "USD"
        if "EUR" in full_text or "€" in full_text:
            currency = "EUR"
        elif "GBP" in full_text or "£" in full_text:
            currency = "GBP"

        # 5. Financial totals
        def extract_amount(patterns: List[str]) -> Optional[Decimal]:
            for p in patterns:
                m = re.search(p, full_text, re.IGNORECASE)
                if m:
                    try:
                        clean_num = m.group(1).replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
                        return Decimal(clean_num)
                    except Exception:
                        continue
            return None

        subtotal = extract_amount([
            r"sub\s*total\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
            r"subtotal\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
        ])

        tax = extract_amount([
            r"(?:tax|sales\s*tax|vat)(?:\s*\([^\)]*\))?\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
        ]) or Decimal("0.00")

        discount = extract_amount([
            r"(?:discount|credit)\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
        ]) or Decimal("0.00")

        freight = extract_amount([
            r"(?:freight|shipping|handling|fuel\s*surcharge|surcharges?)\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
        ]) or Decimal("0.00")

        total = extract_amount([
            r"(?:total\s*due|grand\s*total|amount\s*due)\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
            r"(?<!sub)\btotal\b\s*[:\s]*\$?([0-9,]+\.[0-9]{2})",
        ])
        if total is None:
            missing.append("total_amount")

        # 6. Date & Address
        date_match = re.search(r"(?:date|invoice\s*date)\s*[:\s]*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})", full_text, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None

        addr_match = re.search(r"(?:address|addr)\s*[:\s]*([^\n]+)", full_text, re.IGNORECASE)
        vendor_address = addr_match.group(1).strip() if addr_match else None

        # 7. Extract Line Items from reconstructed visual rows
        extracted_lines: List[LineItem] = []
        line_item_regex = re.compile(
            r"^\s*(?:(\d{1,3})\s+)?([A-Za-z0-9\-_\s]{3,40})\s+(\d+(?:\.\d+)?)\s+\$?([0-9,]+\.[0-9]{2})\s+\$?([0-9,]+\.[0-9]{2})",
            re.IGNORECASE,
        )

        line_counter = 1
        for row in lines:
            m = line_item_regex.match(row)
            if m:
                desc = m.group(2).strip()
                if any(k in desc.lower() for k in ["description", "subtotal", "total", "amount", "unit price", "tax", "freight"]):
                    continue
                try:
                    l_num = int(m.group(1)) if m.group(1) else line_counter
                    qty = Decimal(m.group(3))
                    price = Decimal(m.group(4).replace(",", ""))
                    ltotal = Decimal(m.group(5).replace(",", ""))
                    extracted_lines.append(
                        LineItem(
                            line_number=l_num,
                            description=desc,
                            quantity=qty,
                            unit_price=price,
                            line_total=ltotal,
                        )
                    )
                    line_counter += 1
                except Exception:
                    continue

        if not extracted_lines:
            missing.append("line_items")

        return Invoice(
            invoice_number=inv_number,
            vendor=vendor,
            vendor_address=vendor_address,
            date=date_str,
            po_number=po_number,
            currency=currency,
            lines=extracted_lines,
            subtotal=subtotal,
            tax=tax,
            discount=discount,
            freight_surcharges=freight,
            total=total,
            missing_fields=missing,
        )
