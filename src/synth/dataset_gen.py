"""Synthetic Procurement Dataset Generator.

Generates coordinated test sets:
  - Invoice PDFs (created natively using PyMuPDF)
  - Authorized Purchase Orders (JSON)
  - Warehouse Goods Receipts (JSON)
  - Vendor Contracts / MSAs (JSON)
  - Labeled Ground Truth for evaluation (JSON)
  - Downloadable ZIP archive containing all sample documents

Seeds the 5 most common discrepancy scenarios:
  1. Clean 3-Way Match (Auto-Approve baseline)
  2. Unit Price Variance ($750.00 billed vs $600.00 authorized)
  3. Missing Required Fields / Extraction Incomplete (Human-In-The-Loop trigger)
  4. Cumulative Quantity Overbilling across multi-part invoices
  5. Unauthorized Stealth Surcharges violating Contract Clause
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
import zipfile
import pymupdf

from src.core.schema import GoodsReceipt, PurchaseOrder
from src.core.table_parser import export_goods_receipts_to_csv, export_purchase_orders_to_csv


def draw_invoice_pdf(
    out_path: Path,
    invoice_number: str,
    vendor: str,
    vendor_address: str,
    date: str,
    po_number: str,
    lines: List[Dict[str, Any]],
    subtotal: Decimal,
    tax: Decimal,
    freight: Decimal,
    total: Decimal,
    omit_fields: bool = False,
) -> None:
    """Render a clean, standard procurement invoice PDF using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # Standard A4 size

    # Header section
    y = 50
    page.insert_text((50, y), "COMMERCIAL INVOICE", fontsize=18, color=(0.1, 0.2, 0.5))
    y += 28

    if not omit_fields:
        page.insert_text((50, y), f"Vendor: {vendor}", fontsize=11, color=(0, 0, 0))
        page.insert_text((350, y), f"Invoice #: {invoice_number}", fontsize=11, color=(0, 0, 0))
        y += 16
        page.insert_text((50, y), f"Address: {vendor_address}", fontsize=9, color=(0.3, 0.3, 0.3))
        page.insert_text((350, y), f"Date: {date}", fontsize=10, color=(0.3, 0.3, 0.3))
        y += 16
        page.insert_text((50, y), "Currency: USD", fontsize=10, color=(0.3, 0.3, 0.3))
        page.insert_text((350, y), f"PO Number: {po_number}", fontsize=10, color=(0.3, 0.3, 0.3))
    else:
        # Deliberately omit required fields to trigger Human-in-the-Loop review
        page.insert_text((50, y), "Notice: Vendor copy - Incomplete Header", fontsize=10, color=(0.5, 0.1, 0.1))

    y += 32
    page.draw_line((50, y), (545, y), color=(0.8, 0.8, 0.8), width=1)
    y += 20

    # Table Header
    page.insert_text((50, y), "Line", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((90, y), "Description", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((300, y), "Qty", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((380, y), "Unit Price", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((470, y), "Total", fontsize=10, color=(0.3, 0.3, 0.3))
    y += 15
    page.draw_line((50, y), (545, y), color=(0.8, 0.8, 0.8), width=1)
    y += 18

    # Line items
    for item in lines:
        page.insert_text((50, y), str(item["line"]), fontsize=10)
        page.insert_text((90, y), str(item["desc"]), fontsize=10)
        page.insert_text((300, y), str(item["qty"]), fontsize=10)
        page.insert_text((380, y), f"${item['price']:.2f}", fontsize=10)
        page.insert_text((470, y), f"${item['total']:.2f}", fontsize=10)
        y += 20

    y += 15
    page.draw_line((350, y), (545, y), color=(0.8, 0.8, 0.8), width=1)
    y += 20

    # Totals
    page.insert_text((350, y), "Subtotal:", fontsize=10)
    page.insert_text((470, y), f"${subtotal:.2f}", fontsize=10)
    y += 18

    if freight > Decimal("0.00"):
        page.insert_text((350, y), "Freight / Surcharges:", fontsize=10, color=(0.7, 0.1, 0.1))
        page.insert_text((470, y), f"${freight:.2f}", fontsize=10, color=(0.7, 0.1, 0.1))
        y += 18

    page.insert_text((350, y), "Sales Tax (5%):", fontsize=10)
    page.insert_text((470, y), f"${tax:.2f}", fontsize=10)
    y += 20

    page.draw_line((350, y), (545, y), color=(0.2, 0.2, 0.2), width=1.5)
    y += 20
    page.insert_text((350, y), "TOTAL DUE:", fontsize=12, color=(0.1, 0.2, 0.5))
    page.insert_text((470, y), f"${total:.2f}", fontsize=12, color=(0.1, 0.2, 0.5))

    doc.save(str(out_path))
    doc.close()


def create_sample_zip(data_dir: Path) -> Path:
    """Package all sample procurement benchmark files into a downloadable ZIP archive."""
    zip_path = data_dir / "sample_procurement_documents.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Invoice PDFs
        pdf_dir = data_dir / "invoices"
        if pdf_dir.exists():
            for pdf_file in sorted(pdf_dir.glob("*.pdf")):
                zf.write(pdf_file, arcname=f"invoices/{pdf_file.name}")
        # 2. Master Table & Agreement datasets (Orders in CSV file format & JSON)
        for fname in [
            "purchase_orders.csv",
            "purchase_orders.json",
            "goods_receipts.csv",
            "goods_receipts.json",
            "contracts.json",
            "manifest.json",
        ]:
            fpath = data_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
        # 3. Ground truth evaluation labels
        gt_dir = data_dir / "ground_truth"
        if gt_dir.exists():
            for gt_file in sorted(gt_dir.glob("*.json")):
                zf.write(gt_file, arcname=f"ground_truth/{gt_file.name}")
        # 4. Instructions Readme
        readme_text = (
            "===========================================================\n"
            "PROCUREMENT INVOICE & CONTRACT AUDITOR - SAMPLE DOCUMENTS\n"
            "===========================================================\n\n"
            "This archive contains verified procurement benchmark documents with\n"
            "proper chronological date sequences, physical addresses, and ground truth labels:\n\n"
            "1. invoices/\n"
            "   - INV-101_Clean_Approved.pdf: Clean 3-way match against PO-2026-001\n"
            "   - INV-102_Price_Variance.pdf: Unit price drift from $600 to $750 (PO-2026-002)\n"
            "   - INV-103_Missing_Fields_HITL.pdf: Scanned / missing header fields triggering HITL\n"
            "   - INV-105_Contract_Violation_Surcharge.pdf: Banned freight fee violation (PO-2026-001)\n\n"
            "2. ERP Order Documents (CSV File Format):\n"
            "   - purchase_orders.csv:\n"
            "     Relational table of authorized Purchase Orders with addresses and issuance dates.\n"
            "   - goods_receipts.csv:\n"
            "     Warehouse physical receipt proof table of good orders with verified delivery dates.\n\n"
            "3. contracts.json:\n"
            "   - Master Service Agreement (MSA) legal clauses indexed for Vector RAG.\n\n"
            "4. ground_truth/:\n"
            "   - Expected decisions and reason codes for deterministic benchmark validation.\n\n"
            "To audit, upload files in the Web Cockpit or via POST /api/session/start\n"
        )
        zf.writestr("README.txt", readme_text)
    return zip_path


def generate_synthetic_dataset(output_dir: Path) -> Dict[str, Any]:
    """Generate complete synthetic dataset with seeded discrepancies, proper dates, and ground-truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "invoices"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard Master Purchase Orders with Proper Issuance Dates and Addresses
    purchase_orders = [
        {
            "po_number": "PO-2026-001",
            "vendor": "Apex Industrial Supply",
            "vendor_address": "100 Industrial Parkway, Suite 400, Chicago, IL 60601",
            "date": "2026-10-10",
            "currency": "USD",
            "receipt_required": True,
            "max_tolerated_price_variance": "0.00",
            "lines": [
                {"line_number": 1, "sku": "SKU-BALL-VALVE", "description": "Precision Ball Valve 2in", "quantity": "10.0", "unit_price": "50.00"},
                {"line_number": 2, "sku": "SKU-SEAL-KIT", "description": "EPDM Gasket Seal Kit", "quantity": "20.0", "unit_price": "15.00"},
            ],
        },
        {
            "po_number": "PO-2026-002",
            "vendor": "Nexus Cloud Hardware",
            "vendor_address": "750 Innovation Way, Austin, TX 78701",
            "date": "2026-10-12",
            "currency": "USD",
            "receipt_required": True,
            "max_tolerated_price_variance": "0.00",
            "lines": [
                {"line_number": 1, "sku": "SKU-SERVER-RACK", "description": "42U Server Rack Cabinet", "quantity": "4.0", "unit_price": "600.00"},
            ],
        },
        {
            "po_number": "PO-2026-003",
            "vendor": "Vanguard Logistics & Parts",
            "vendor_address": "520 Commerce Blvd, Atlanta, GA 30303",
            "date": "2026-10-14",
            "currency": "USD",
            "receipt_required": True,
            "max_tolerated_price_variance": "0.00",
            "lines": [
                {"line_number": 1, "sku": "SKU-STEEL-BRACKET", "description": "Reinforced Steel Bracket", "quantity": "100.0", "unit_price": "10.00"},
            ],
        },
    ]

    # 2. Master Goods Receipts (Chronologically after PO issuance)
    goods_receipts = [
        {
            "receipt_id": "GR-2026-001",
            "po_number": "PO-2026-001",
            "receipt_date": "2026-10-18",
            "lines": [
                {"line_number": 1, "sku": "SKU-BALL-VALVE", "quantity_received": "10.0"},
                {"line_number": 2, "sku": "SKU-SEAL-KIT", "quantity_received": "20.0"},
            ],
        },
        {
            "receipt_id": "GR-2026-002",
            "po_number": "PO-2026-002",
            "receipt_date": "2026-10-19",
            "lines": [
                {"line_number": 1, "sku": "SKU-SERVER-RACK", "quantity_received": "4.0"},
            ],
        },
        {
            "receipt_id": "GR-2026-003",
            "po_number": "PO-2026-003",
            "receipt_date": "2026-10-20",
            "lines": [
                {"line_number": 1, "sku": "SKU-STEEL-BRACKET", "quantity_received": "100.0"},
            ],
        },
    ]

    # 3. Master Contract Clauses (Effective prior to POs)
    contract_clauses = [
        {
            "clause_id": "MSA-APEX-SEC-4.2",
            "vendor": "Apex Industrial Supply",
            "title": "All-Inclusive Freight Policy",
            "content": "All orders under PO include standard delivery. Vendor shall not bill separate freight, delivery fees, or fuel surcharges.",
            "category": "fees",
            "prohibits_surcharges": True,
            "allowed_payment_terms": "Net 30",
            "discount_percentage": None,
        },
        {
            "clause_id": "MSA-NEXUS-SEC-9.1",
            "vendor": "Nexus Cloud Hardware",
            "title": "Fixed Rate Card Guarantee",
            "content": "Prices stated in PO are fixed for 12 months. Any price escalation without prior written authorization is strictly void.",
            "category": "pricing",
            "prohibits_surcharges": False,
            "allowed_payment_terms": "Net 30",
            "discount_percentage": None,
        },
    ]

    # 4. Benchmark Invoices (Chronologically after Goods Receipt)
    scenarios = [
        # 1. Clean Auto-Approve Baseline
        {
            "id": "INV-101",
            "pdf_name": "INV-101_Clean_Approved.pdf",
            "vendor": "Apex Industrial Supply",
            "vendor_address": "100 Industrial Parkway, Suite 400, Chicago, IL 60601",
            "date": "2026-10-22",
            "po": "PO-2026-001",
            "lines": [
                {"line": 1, "desc": "Precision Ball Valve 2in", "qty": Decimal("10.0"), "price": Decimal("50.00"), "total": Decimal("500.00")},
                {"line": 2, "desc": "EPDM Gasket Seal Kit", "qty": Decimal("20.0"), "price": Decimal("15.00"), "total": Decimal("300.00")},
            ],
            "subtotal": Decimal("800.00"),
            "tax": Decimal("40.00"),
            "freight": Decimal("0.00"),
            "total": Decimal("840.00"),
            "omit": False,
            "expected_decision": "auto_approve",
            "expected_reasons": [],
            "description": "Clean 3-way match against PO and warehouse receipts.",
        },
        # 2. Price Variance Overcharge
        {
            "id": "INV-102",
            "pdf_name": "INV-102_Price_Variance.pdf",
            "vendor": "Nexus Cloud Hardware",
            "vendor_address": "750 Innovation Way, Austin, TX 78701",
            "date": "2026-10-23",
            "po": "PO-2026-002",
            "lines": [
                {"line": 1, "desc": "42U Server Rack Cabinet", "qty": Decimal("4.0"), "price": Decimal("750.00"), "total": Decimal("3000.00")},  # PO was $600
            ],
            "subtotal": Decimal("3000.00"),
            "tax": Decimal("150.00"),
            "freight": Decimal("0.00"),
            "total": Decimal("3150.00"),
            "omit": False,
            "expected_decision": "needs_review",
            "expected_reasons": ["PRICE_VARIANCE"],
            "description": "Unit price inflated from $600.00 to $750.00 per unit ($600 overcharge).",
        },
        # 3. Missing Required Field / Human-In-The-Loop Trigger
        {
            "id": "INV-103",
            "pdf_name": "INV-103_Missing_Fields_HITL.pdf",
            "vendor": "Unknown Supplier",
            "vendor_address": "",
            "date": "",
            "po": "PO-2026-001",
            "lines": [],
            "subtotal": Decimal("0.00"),
            "tax": Decimal("0.00"),
            "freight": Decimal("0.00"),
            "total": Decimal("0.00"),
            "omit": True,
            "expected_decision": "needs_review",
            "expected_reasons": ["MISSING_REQUIRED_FIELD"],
            "description": "Omitted vendor and line items to trigger Human-In-The-Loop review.",
        },
        # 4. Unauthorized Contract Surcharge
        {
            "id": "INV-105",
            "pdf_name": "INV-105_Contract_Violation_Surcharge.pdf",
            "vendor": "Apex Industrial Supply",
            "vendor_address": "100 Industrial Parkway, Suite 400, Chicago, IL 60601",
            "date": "2026-10-22",
            "po": "PO-2026-001",
            "lines": [
                {"line": 1, "desc": "Precision Ball Valve 2in", "qty": Decimal("10.0"), "price": Decimal("50.00"), "total": Decimal("500.00")},
                {"line": 2, "desc": "EPDM Gasket Seal Kit", "qty": Decimal("20.0"), "price": Decimal("15.00"), "total": Decimal("300.00")},
            ],
            "subtotal": Decimal("800.00"),
            "tax": Decimal("40.00"),
            "freight": Decimal("150.00"),  # Prohibited by MSA-APEX-SEC-4.2
            "total": Decimal("990.00"),
            "omit": False,
            "expected_decision": "reject",
            "expected_reasons": ["UNAUTHORIZED_SURCHARGE"],
            "description": "$150 unauthorized freight surcharge strictly banned by Master Contract.",
        },
    ]

    # Write files
    for sc in scenarios:
        pdf_path = pdf_dir / sc["pdf_name"]
        draw_invoice_pdf(
            out_path=pdf_path,
            invoice_number=sc["id"],
            vendor=sc["vendor"],
            vendor_address=sc.get("vendor_address", ""),
            date=sc.get("date", ""),
            po_number=sc["po"],
            lines=sc["lines"],
            subtotal=sc["subtotal"],
            tax=sc["tax"],
            freight=sc["freight"],
            total=sc["total"],
            omit_fields=sc["omit"],
        )

        # Ground truth label
        gt_data = {
            "invoice_id": sc["id"],
            "expected_decision": sc["expected_decision"],
            "expected_reasons": sc["expected_reasons"],
            "description": sc["description"],
            "ground_truth_total": str(sc["total"]),
        }
        with open(gt_dir / f"{sc['id']}_gt.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

    # Save shared datasets (both tabular CSV and JSON)
    po_objs = [PurchaseOrder(**p) for p in purchase_orders]
    gr_objs = [GoodsReceipt(**g) for g in goods_receipts]

    with open(output_dir / "purchase_orders.csv", "w", encoding="utf-8") as f:
        f.write(export_purchase_orders_to_csv(po_objs))

    with open(output_dir / "purchase_orders.json", "w", encoding="utf-8") as f:
        json.dump(purchase_orders, f, indent=2)

    with open(output_dir / "goods_receipts.csv", "w", encoding="utf-8") as f:
        f.write(export_goods_receipts_to_csv(gr_objs))

    with open(output_dir / "goods_receipts.json", "w", encoding="utf-8") as f:
        json.dump(goods_receipts, f, indent=2)

    with open(output_dir / "contracts.json", "w", encoding="utf-8") as f:
        json.dump(contract_clauses, f, indent=2)

    manifest = {
        "dataset_name": "procurement_benchmark_v1",
        "invoices_count": len(scenarios),
        "invoices": [
            {
                "id": s["id"],
                "file": s["pdf_name"],
                "vendor": s["vendor"],
                "po_number": s["po"],
                "expected_decision": s["expected_decision"],
            }
            for s in scenarios
        ],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Package everything into downloadable ZIP
    create_sample_zip(output_dir)

    return manifest
