"""Entity name, address, and document normalization helpers.

Handles corporate suffix ignorance (e.g. 'ABC', 'ABC Inc', 'ABC LLC' resolve to the same entity).
Builds composite unique history keys incorporating Name + Address + Invoice ID + Date.
"""

import re
from typing import Optional

# Common corporate legal suffixes to ignore for entity comparison
CORPORATE_SUFFIXES = [
    r"\bincorporated\b", r"\binc\.?\b",
    r"\bcorporation\b", r"\bcorp\.?\b",
    r"\bl\.?l\.?c\.?\b", r"\bllc\b",
    r"\blimited\b", r"\bltd\.?\b",
    r"\bcompany\b", r"\bco\.?\b",
    r"\bgmbh\b", r"\bplc\b",
    r"\bgroup\b", r"\bholdings?\b",
]

# Common address abbreviations to normalize
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\bave\b": "avenue",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bpkwy\b": "parkway",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
}


def normalize_entity_name(name: Optional[str]) -> str:
    """Normalize supplier/entity name ignoring corporate legal suffixes and formatting.
    
    Examples:
        'ABC Inc.' -> 'abc'
        'ABC' -> 'abc'
        'ABC LLC' -> 'abc'
        'Apex Industrial Supply, Inc.' -> 'apex industrial supply'
    """
    if not name:
        return ""

    # 1. Lowercase and normalize ampersands
    norm = name.lower().replace("&", " and ")

    # 2. Remove punctuation characters
    norm = re.sub(r"[,\.\-\'\/\(\)\"]", " ", norm)

    # 3. Strip corporate suffixes
    for suffix in CORPORATE_SUFFIXES:
        norm = re.sub(suffix, " ", norm, flags=re.IGNORECASE)

    # 4. Collapse extra whitespace
    return " ".join(norm.split())


def normalize_address(address: Optional[str]) -> str:
    """Normalize postal / street addresses for cross-document deduplication."""
    if not address:
        return ""

    # Lowercase & strip punctuation
    norm = address.lower()
    norm = re.sub(r"[,\.\-\#\/\(\)\"]", " ", norm)

    # Standardize street abbreviations
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        norm = re.sub(pattern, replacement, norm, flags=re.IGNORECASE)

    return " ".join(norm.split())


def build_unique_history_key(
    vendor: Optional[str],
    invoice_number: Optional[str],
    date: Optional[str] = None,
    address: Optional[str] = None,
) -> str:
    """Construct composite history fingerprint combining Name + ID + Date + Address.
    
    Provides high-fidelity unique history verification so legitimate re-billings
    or differing physical entities are correctly distinguished.
    """
    v_norm = normalize_entity_name(vendor)
    id_norm = (invoice_number or "").strip().lower()
    date_norm = (date or "").strip()
    addr_norm = normalize_address(address)

    parts = [v_norm, id_norm]
    if date_norm:
        parts.append(date_norm)
    if addr_norm:
        parts.append(addr_norm)

    return "|".join(parts)
