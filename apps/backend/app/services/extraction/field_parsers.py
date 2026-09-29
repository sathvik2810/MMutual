"""Semantic and pattern-based parsing of OCR text."""
from __future__ import annotations
import re

_PATTERNS: dict[str, re.Pattern] = {
    "date": re.compile(r"(?:Date|Dated)\s*[:\-]?\s*((?:[0-9]{1,4}[/\-.][0-9]{1,2}[/\-.][0-9]{1,4})|(?:[0-9]{1,2}[-/ ]+[A-Za-z]{3,9}[-/ ]+[0-9]{2,4})|(?:[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}))", re.I),
    "cheque_number": re.compile(r"(?:Cheque|Check|Chq)\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9]{4,12})", re.I),
    "account_number": re.compile(r"(?:Account|A\s*/?\s*C)\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9]{6,24})", re.I),
    "routing_transit_number": re.compile(r"(?:Routing\s*/?\s*Transit|Routing|IFSC)\s*(?:No\.?|Code)?\s*[:\-]?\s*([A-Za-z0-9]{6,16})", re.I),
    "amount": re.compile(r"(?:[$]|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)(?:\s*/-)?", re.I),
    "amount_in_words": re.compile(r"(?:Amount\s+in\s+words|Rupees)\s*[:\-]?\s*(.+)", re.I),
    "payee_name": re.compile(r"(?:Pay\s+to(?:\s+the\s+order\s+of)?|Payee)\s*[:\-]?\s*(.+)", re.I),
}

_UNLABELLED_DATE = re.compile(
    r"(?<!\d)(?:[0-9]{1,4}[/\-.][0-9]{1,2}[/\-.][0-9]{1,4}|"
    r"[0-9]{1,2}[-/ ]+[A-Za-z]{3,9}[-/ ]+[0-9]{2,4}|"
    r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})(?!\d)",
    re.I,
)

def parse_field_from_text(field_name: str, raw_text: str) -> str | None:
    pattern = _PATTERNS.get(field_name)
    match = pattern.search(raw_text) if pattern and raw_text else None
    if not match:
        return None
    value = match.group(1).strip()
    if field_name in {"amount_in_words", "payee_name"}:
        value = value.splitlines()[0].strip()
    return value or None


def parse_unlabelled_date(raw_text: str) -> str | None:
    """Return a date only when OCR text contains one unique valid candidate."""
    from app.services.extraction.normalization import normalize_date

    candidates = []
    for match in _UNLABELLED_DATE.finditer(raw_text or ""):
        candidate = match.group(0).strip()
        normalized = normalize_date(candidate)
        if normalized is not None:
            candidates.append((candidate, normalized))
    distinct_dates = {normalized for _, normalized in candidates}
    if len(distinct_dates) != 1:
        return None
    return next(candidate for candidate, normalized in candidates if normalized in distinct_dates)

def strip_known_label(field_name: str, text: str) -> str:
    pattern = _PATTERNS.get(field_name)
    match = pattern.match(text) if pattern and text else None
    if not match:
        return text
    value = match.group(1).strip()
    if field_name in {"amount_in_words", "payee_name"}:
        value = value.splitlines()[0].strip()
    return value

def parse_bank_name(raw_text: str) -> str | None:
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return None
