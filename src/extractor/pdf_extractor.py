"""Dynamic PDF Extractor combining PyMuPDF text reconstruction with LLM understanding.

Handles custom, arbitrary PDF invoice layouts without hardcoding.
Explicitly detects scanned PDFs (no text layer) and missing required fields,
triggering Human-in-the-Loop review.
"""

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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

        # Dynamic Extraction via LLM (with fallback to semantic lowercase parser)
        extracted = llm_client.extract_invoice_dynamic(reconstructed_text)

        # Build clean Pydantic model
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

        invoice = Invoice(
            invoice_number=inv_num,
            vendor=vendor,
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

        return invoice, reasons
