"""Dynamic procurement table and clause parser for ERP Orders and Contract Agreements.

Features:
- Dynamic schema detection: Operates independently of file names (inspects structure & content).
- Dual tabular order format support: CSV, TSV, and JSON for Purchase Orders and Good Orders (Goods Receipts).
- Dedicated Contract Clause parser: Separates legal agreements (CSV tables, JSON, Markdown/Text) from order tables.
- Robust header synonym mapping: Ingests real-world heterogeneous ERP column names without strict naming conventions.
- Decimal precision: All financial rates, subtotals, and physical quantities strictly map to decimal.Decimal.
"""

import csv
import io
import json
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.schema import ContractClause, GoodsReceipt, LineItem, POLine, PurchaseOrder, ReceiptLine


def _clean_str(val: Any) -> str:
    """Trim whitespace and return clean string."""
    if val is None:
        return ""
    return str(val).strip()


def _normalize_header(header: str) -> str:
    """Normalize tabular header string (e.g. 'PO Number' -> 'po_number')."""
    return re.sub(r"[^a-z0-9_]", "", header.strip().lower().replace(" ", "_").replace("-", "_"))


def _decode_to_text(content: Union[str, bytes]) -> str:
    """Safely decode bytes or normalize text string."""
    if isinstance(content, bytes):
        return content.decode("utf-8-sig", errors="replace").lstrip("\ufeff")
    return content.lstrip("\ufeff")


# --- Dynamic Document Type Detection ---

def detect_document_type(content: Union[str, bytes], filename: Optional[str] = None) -> str:
    """Dynamically determine whether content represents POs, Good Orders (Receipts), or Contract Clauses.
    
    Operates independently of input file name by inspecting column headers, JSON keys, and keywords.
    """
    text = _decode_to_text(content).strip()
    if not text:
        return "contract_clauses"

    # 1. Inspect JSON keys if JSON format
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            sample = data[0] if isinstance(data, list) and len(data) > 0 else data if isinstance(data, dict) else {}
            keys = set(str(k).lower() for k in sample.keys())
            
            # Receipts check
            if any(k in keys for k in ["receipt_id", "gr_number", "quantity_received", "qty_received"]):
                return "goods_receipts"
            # Clause check
            if any(k in keys for k in ["clause_id", "prohibits_surcharges", "allowed_payment_terms"]):
                return "contract_clauses"
            # PO check
            if any(k in keys for k in ["po_number", "unit_price", "purchase_order", "receipt_required"]):
                return "purchase_orders"
        except Exception:
            pass

    # 2. Inspect Tabular Headers (CSV/TSV)
    first_line = text.split("\n")[0].lower()
    delimiter = "\t" if "\t" in first_line else ","
    headers = [_normalize_header(h) for h in first_line.split(delimiter)]

    receipt_indicators = {"receipt_id", "receipt_number", "gr_number", "quantity_received", "qty_received", "delivery_date"}
    po_indicators = {"po_number", "po", "purchase_order", "unit_price", "price", "rate", "receipt_required", "line_total"}
    clause_indicators = {"clause_id", "clause", "prohibits_surcharges", "terms", "clause_title", "penalty"}

    receipt_score = sum(1 for h in headers if h in receipt_indicators)
    po_score = sum(1 for h in headers if h in po_indicators)
    clause_score = sum(1 for h in headers if h in clause_indicators)

    if receipt_score > 0 and receipt_score >= po_score:
        return "goods_receipts"
    if po_score > 0:
        return "purchase_orders"
    if clause_score > 0:
        return "contract_clauses"

    # 3. Fallback: Check for natural legal language / clauses
    if re.search(r"\b(clause|section|article|agreement|msa|surcharge|terms and conditions)\b", text, re.IGNORECASE):
        return "contract_clauses"

    return "purchase_orders"


# --- Dynamic Purchase Order Parser ---

def parse_purchase_orders(
    content: Union[str, bytes],
    filename: Optional[str] = None,
) -> Dict[str, PurchaseOrder]:
    """Parse Purchase Orders from CSV, TSV, or JSON table content dynamically.
    
    Independent of proper file name. Returns Dict[po_number -> PurchaseOrder].
    """
    text = _decode_to_text(content).strip()
    if not text:
        return {}

    # JSON handling
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            orders: Dict[str, PurchaseOrder] = {}
            for item in data:
                if "lines" in item and isinstance(item["lines"], list):
                    po = PurchaseOrder(**item)
                    orders[po.po_number] = po
                else:
                    return _group_po_rows(data)
            return orders
        except json.JSONDecodeError:
            pass

    # CSV/TSV handling
    delimiter = "\t" if "\t" in text.split("\n")[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    
    rows: List[Dict[str, str]] = []
    for r in reader:
        norm_row = {_normalize_header(k): _clean_str(v) for k, v in r.items() if k is not None}
        rows.append(norm_row)

    return _group_po_rows(rows)


def _find_field(row: Dict[str, Any], synonyms: List[str], default: Any = None) -> Any:
    """Find value in normalized row matching any synonym substring in keys."""
    for syn in synonyms:
        if syn in row and row[syn] not in (None, ""):
            return row[syn]
    for syn in synonyms:
        for k, v in row.items():
            if syn in k and v not in (None, ""):
                return v
    return default


def _group_po_rows(rows: List[Dict[str, Any]]) -> Dict[str, PurchaseOrder]:
    """Group flat tabular rows into PurchaseOrder domain models with line items."""
    grouped: Dict[str, Dict[str, Any]] = {}
    lines_by_po: Dict[str, List[POLine]] = {}

    for row in rows:
        po_number = _find_field(
            row,
            ["po_number", "ponum", "po_num", "po_no", "pono", "purchase_order", "order_number", "orderno", "order_id", "order", "po"],
            "",
        )
        if not po_number:
            continue

        if po_number not in grouped:
            vendor = _find_field(
                row,
                ["vendor_name", "vendor", "supplier", "seller", "company", "contractor"],
                "Unknown Vendor",
            )
            vendor_address = _find_field(
                row,
                ["vendor_address", "street_address", "street", "address", "addr", "location"],
                None,
            )
            date = _find_field(
                row,
                ["po_date", "order_date", "issue_date", "creation_date", "date"],
                None,
            )
            currency = _find_field(row, ["currency", "curr"], "USD")
            
            raw_rec = str(_find_field(row, ["receipt_required", "require_receipt", "3way"], "true")).lower()
            receipt_required = raw_rec in ("true", "1", "yes", "y", "t")
            
            raw_tol = _find_field(row, ["max_tolerated_price_variance", "price_variance_tolerance", "variance_tol"], "0.00")
            try:
                max_tol = Decimal(str(raw_tol).replace("$", "").strip() or "0.00")
            except Exception:
                max_tol = Decimal("0.00")

            grouped[po_number] = {
                "po_number": po_number,
                "vendor": vendor,
                "vendor_address": vendor_address,
                "date": date,
                "currency": currency,
                "receipt_required": receipt_required,
                "max_tolerated_price_variance": max_tol,
            }
            lines_by_po[po_number] = []

        # Parse Line Item
        raw_line_num = _find_field(row, ["line_number", "lineno", "line_no", "line_id", "line"], None)
        try:
            line_num = int(raw_line_num) if raw_line_num else len(lines_by_po[po_number]) + 1
        except Exception:
            line_num = len(lines_by_po[po_number]) + 1

        sku = _find_field(row, ["sku", "item_sku", "part_number", "part_no", "partno", "item_code", "part"], None)
        description = _find_field(row, ["item_description", "description", "item_desc", "item", "desc", "product", "title"], f"Line {line_num}")
        
        raw_qty = _find_field(row, ["ordered_quantity", "quantity", "qty", "count", "amount"], "1.0")
        raw_price = _find_field(row, ["unit_price", "unitprice", "unit_cost", "rate", "price", "cost"], "0.00")
        raw_tax = _find_field(row, ["tax_rate", "tax_pct", "tax"], "0.00")

        pol = POLine(
            line_number=line_num,
            sku=sku,
            description=description,
            quantity=raw_qty,
            unit_price=raw_price,
            tax_rate=Decimal(str(raw_tax).replace("%", "").strip() or "0.00"),
        )
        lines_by_po[po_number].append(pol)

    orders: Dict[str, PurchaseOrder] = {}
    for po_num, base in grouped.items():
        base["lines"] = lines_by_po.get(po_num, [])
        orders[po_num] = PurchaseOrder(**base)

    return orders


# --- Dynamic Good Orders / Goods Receipts Parser ---

def parse_goods_receipts(
    content: Union[str, bytes],
    filename: Optional[str] = None,
) -> Dict[str, GoodsReceipt]:
    """Parse Good Orders / Goods Receipts from CSV, TSV, or JSON table content dynamically.
    
    Independent of proper file name. Returns Dict[po_number -> GoodsReceipt].
    """
    text = _decode_to_text(content).strip()
    if not text:
        return {}

    # JSON handling
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            receipts: Dict[str, GoodsReceipt] = {}
            for item in data:
                if "lines" in item and isinstance(item["lines"], list):
                    gr = GoodsReceipt(**item)
                    receipts[gr.po_number] = gr
                else:
                    return _group_receipt_rows(data)
            return receipts
        except json.JSONDecodeError:
            pass

    # CSV/TSV handling
    delimiter = "\t" if "\t" in text.split("\n")[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    rows: List[Dict[str, str]] = []
    for r in reader:
        norm_row = {_normalize_header(k): _clean_str(v) for k, v in r.items() if k is not None}
        rows.append(norm_row)

    return _group_receipt_rows(rows)


def _group_receipt_rows(rows: List[Dict[str, Any]]) -> Dict[str, GoodsReceipt]:
    """Group flat tabular receipt rows into GoodsReceipt domain models with lines."""
    grouped: Dict[str, Dict[str, Any]] = {}
    lines_by_po: Dict[str, List[ReceiptLine]] = {}

    for row in rows:
        po_number = _find_field(
            row,
            ["po_number", "ponum", "po_no", "pono", "order_number", "orderno", "order_id", "order", "po"],
            "",
        )
        if not po_number:
            continue

        receipt_id = _find_field(
            row,
            ["receipt_id", "receipt_number", "receiptno", "gr_number", "gr_no", "grno", "packing_slip", "delivery_id", "receipt"],
            f"GR-{po_number}",
        )
        receipt_date = _find_field(
            row,
            ["receipt_date", "delivery_date", "received_date", "delivery", "date"],
            "2026-01-01",
        )

        if po_number not in grouped:
            grouped[po_number] = {
                "receipt_id": receipt_id,
                "po_number": po_number,
                "receipt_date": receipt_date,
            }
            lines_by_po[po_number] = []

        raw_line_num = _find_field(row, ["line_number", "lineno", "line_no", "line_id", "line"], None)
        try:
            line_num = int(raw_line_num) if raw_line_num else len(lines_by_po[po_number]) + 1
        except Exception:
            line_num = len(lines_by_po[po_number]) + 1

        sku = _find_field(row, ["sku", "item_sku", "part_number", "part_no", "partno", "item_code", "part"], None)
        raw_qty = _find_field(
            row,
            ["delivered_qty", "delivered_quantity", "quantity_received", "qty_received", "received_quantity", "received_qty", "received", "qty", "quantity", "count"],
            "1.0",
        )

        rl = ReceiptLine(
            line_number=line_num,
            sku=sku,
            quantity_received=raw_qty,
        )
        lines_by_po[po_number].append(rl)

    receipts: Dict[str, GoodsReceipt] = {}
    for po_num, base in grouped.items():
        base["lines"] = lines_by_po.get(po_num, [])
        receipts[po_num] = GoodsReceipt(**base)

    return receipts


# --- Separate Dynamic Contract Clause Parser ---

def parse_contract_clauses(
    content: Union[str, bytes],
    filename: Optional[str] = None,
) -> List[ContractClause]:
    """Dynamically parse legal clauses from CSV tables, JSON arrays, or Markdown/Text files.
    
    Independent of proper file name. Separates legal compliance terms from operational orders.
    """
    text = _decode_to_text(content).strip()
    if not text:
        return []

    # 1. JSON Array / Object Parsing
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("clauses", [data])
            clauses: List[ContractClause] = []
            for item in data:
                if isinstance(item, dict):
                    # Normalize keys
                    clause_id = item.get("clause_id") or item.get("id") or f"CLAUSE-{len(clauses)+1}"
                    vendor = item.get("vendor") or item.get("supplier") or "Contractor"
                    title = item.get("title") or item.get("heading") or "Contract Terms"
                    c_text = item.get("content") or item.get("text") or item.get("clause_text") or ""
                    prohibits = bool(item.get("prohibits_surcharges", False))
                    category = item.get("category") or "pricing"
                    discount = item.get("discount_percentage")
                    disc_dec = Decimal(str(discount).replace("%", "")) if discount is not None else None

                    clauses.append(ContractClause(
                        clause_id=clause_id,
                        vendor=vendor,
                        title=title,
                        content=c_text,
                        category=category,
                        prohibits_surcharges=prohibits,
                        discount_percentage=disc_dec,
                    ))
            if clauses:
                return clauses
        except json.JSONDecodeError:
            pass

    # 2. Tabular CSV/TSV Parsing
    first_line = text.split("\n")[0].lower()
    if ("clause_id" in first_line or "prohibits_surcharges" in first_line or "title" in first_line) and ("," in first_line or "\t" in first_line):
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        clauses = []
        for idx, r in enumerate(reader, start=1):
            norm_r = {_normalize_header(k): _clean_str(v) for k, v in r.items() if k}
            clause_id = norm_r.get("clause_id") or norm_r.get("id") or f"CLAUSE-{idx}"
            vendor = norm_r.get("vendor") or norm_r.get("supplier") or "Contractor"
            title = norm_r.get("title") or norm_r.get("heading") or f"Section {idx}"
            c_text = norm_r.get("content") or norm_r.get("text") or norm_r.get("clause_text") or ""
            raw_proh = norm_r.get("prohibits_surcharges", "false").lower()
            prohibits = raw_proh in ("true", "1", "yes", "y")
            category = norm_r.get("category") or "pricing"
            raw_disc = norm_r.get("discount_percentage")
            disc_dec = Decimal(str(raw_disc).replace("%", "")) if raw_disc else None

            clauses.append(ContractClause(
                clause_id=clause_id,
                vendor=vendor,
                title=title,
                content=c_text,
                category=category,
                prohibits_surcharges=prohibits,
                discount_percentage=disc_dec,
            ))
        if clauses:
            return clauses

    # 3. Plain Text / Markdown Document Parsing
    # Split text into sections or clauses dynamically
    clauses = []
    # Match patterns like "Clause 1:", "Section 2:", "Article 3:", or markdown headings "# "
    section_splits = re.split(r"(?:\n|^)(?:#+\s*|(?:Section|Clause|Article)\s+[\d\.]+:?\s*)", text, flags=re.IGNORECASE)
    
    clean_sections = [s.strip() for s in section_splits if s and len(s.strip()) > 20]
    if not clean_sections:
        clean_sections = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 20]
    if not clean_sections:
        clean_sections = [text.strip()]

    for idx, sec in enumerate(clean_sections, start=1):
        first_line = sec.split("\n")[0][:80]
        # Check if section forbids stealth surcharges or fuel/freight fees
        prohibits_surcharge = bool(re.search(
            r"\b(separate\s+freight|fuel\s+surcharge|handling\s+fee|unauthorized\s+surcharge|all-inclusive|strictly\s+prohibited)\b",
            sec,
            re.IGNORECASE,
        ))
        clauses.append(ContractClause(
            clause_id=f"DOC-CLAUSE-{idx}",
            vendor="Contractor",
            title=first_line if len(first_line) > 5 else f"Contract Terms Section {idx}",
            content=sec,
            category="pricing" if prohibits_surcharge else "terms",
            prohibits_surcharges=prohibits_surcharge,
        ))

    return clauses


# --- Unified Dynamic Procurement Document Ingestor ---

def parse_procurement_document(
    content: Union[str, bytes],
    doc_type_hint: Optional[str] = None,
    filename: Optional[str] = None,
) -> Tuple[str, Any]:
    """Dynamic universal entrypoint: classifies and parses any procurement document.
    
    Independent of filename.
    Returns:
      ("purchase_orders", Dict[str, PurchaseOrder])
      ("goods_receipts", Dict[str, GoodsReceipt])
      ("contract_clauses", List[ContractClause])
    """
    resolved_type = doc_type_hint or detect_document_type(content, filename=filename)
    if resolved_type == "goods_receipts":
        return "goods_receipts", parse_goods_receipts(content, filename=filename)
    elif resolved_type == "contract_clauses":
        return "contract_clauses", parse_contract_clauses(content, filename=filename)
    else:
        return "purchase_orders", parse_purchase_orders(content, filename=filename)


# --- Relational CSV Export Serializers ---

def export_purchase_orders_to_csv(orders: List[PurchaseOrder]) -> str:
    """Serialize a collection of PurchaseOrders to flat ERP tabular CSV format."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "po_number",
        "vendor",
        "vendor_address",
        "date",
        "currency",
        "receipt_required",
        "line_number",
        "sku",
        "description",
        "quantity",
        "unit_price",
        "tax_rate",
        "max_tolerated_price_variance",
    ])
    for po in orders:
        for line in po.lines:
            writer.writerow([
                po.po_number,
                po.vendor,
                po.vendor_address or "",
                po.date or "",
                po.currency,
                "true" if po.receipt_required else "false",
                line.line_number,
                line.sku or "",
                line.description,
                f"{line.quantity.normalize():f}",
                f"{line.unit_price:.2f}",
                f"{line.tax_rate:.2f}",
                f"{po.max_tolerated_price_variance:.2f}",
            ])
    return output.getvalue()


def export_goods_receipts_to_csv(receipts: List[GoodsReceipt]) -> str:
    """Serialize a collection of GoodsReceipts to flat ERP tabular CSV format."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "receipt_id",
        "po_number",
        "receipt_date",
        "line_number",
        "sku",
        "quantity_received",
    ])
    for gr in receipts:
        for line in gr.lines:
            writer.writerow([
                gr.receipt_id,
                gr.po_number,
                gr.receipt_date,
                line.line_number,
                line.sku or "",
                f"{line.quantity_received.normalize():f}",
            ])
    return output.getvalue()


def export_contract_clauses_to_csv(clauses: List[ContractClause]) -> str:
    """Serialize contract clauses to tabular CSV format."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "clause_id",
        "vendor",
        "title",
        "category",
        "prohibits_surcharges",
        "discount_percentage",
        "content",
    ])
    for c in clauses:
        writer.writerow([
            c.clause_id,
            c.vendor,
            c.title,
            c.category,
            "true" if c.prohibits_surcharges else "false",
            f"{c.discount_percentage:.2f}" if c.discount_percentage is not None else "",
            c.content.replace("\n", " "),
        ])
    return output.getvalue()
