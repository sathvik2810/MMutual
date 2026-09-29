"""Field normalization (docs/15_Cheque_Data_Extraction.md S11-S22).

Each function returns the normalized value or ``None`` when the raw OCR
text cannot be confidently parsed -- normalization never guesses or
fabricates a value it isn't confident about (docs/15 S32: "The module
should never silently replace an uncertain value with a guessed value").
The raw OCR text is always preserved by the caller alongside whatever
these functions return.
"""

from __future__ import annotations

import re
from datetime import datetime

_DIGITS_RE = re.compile(r"\d+")
_DATE_PATTERN_RE = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{1,2}[-/ ]+[A-Za-z]{3,9}[-/ ]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}")
# Contiguous run of digits/commas/dots -- used to locate the numeric
# portion of an amount while ignoring unrelated punctuation elsewhere in
# the string (e.g. the period in the "Rs." abbreviation, which is not
# part of the number itself and must not be swept into it).
_NUMERIC_RUN_RE = re.compile(r"[\d,.]+")

# The Milestone 1 synthetic dataset renders dates as DD/MM/YYYY
# (see scripts/generate_synthetic_data.py); additional common formats are
# accepted defensively, but none are invented -- an unparseable date
# normalizes to None rather than a guess.
_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d",
                 "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
                 "%m/%d/%y", "%m-%d-%y", "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y",
                 "%d %B %Y", "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
                 "%d%m%Y", "%Y%m%d")


def normalize_amount(raw_text: str | None) -> float | None:
    """"$17,334.23" -> 17334.23, "Rs. 25,000/-" -> 25000.0. Returns None
    if no plausible numeric amount can be parsed from the text -- a
    numeric run containing more than one decimal point (e.g. a garbled
    "12.34.56") is treated as unparseable rather than guessing which
    portion is the real number, and a run immediately preceded by "-" is
    rejected outright since a cheque amount cannot be negative."""
    if not raw_text:
        return None

    for match in _NUMERIC_RUN_RE.finditer(raw_text):
        run = match.group(0)
        if not any(ch.isdigit() for ch in run):
            continue
        if run.count(".") > 1:
            return None  # ambiguous malformed number -- do not guess which part is real

        prefix = raw_text[: match.start()].rstrip()
        if prefix.endswith("-"):
            return None  # negative amount is not valid for a cheque

        cleaned = run.replace(",", "")
        if cleaned in (".", ""):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        return round(value, 2)

    return None


def normalize_date(raw_text: str | None) -> str | None:
    """"Date 11/08/2026" -> "2026-08-11" (ISO 8601). The date pattern is
    located within the raw text first (the raw OCR value for this field
    may still include its label, e.g. "Date 11/08/2026", since region
    extraction returns everything OCR found in that area of the cheque),
    then parsed strictly. Returns None -- never a guess -- if no
    date-shaped substring can be found or none of the supported formats
    parse it."""
    if not raw_text:
        return None
    match = _DATE_PATTERN_RE.search(raw_text)
    compact = re.search(r"(?<!\d)\d{8}(?!\d)", raw_text)
    candidate = match.group(0) if match else (compact.group(0) if compact else raw_text.strip())
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def normalize_cheque_number(raw_text: str | None) -> str | None:
    """Preserves leading zeros -- a cheque number is an identifier, not a
    numeric value (docs/15 S12: "000123 must not become 123")."""
    if not raw_text:
        return None
    digits = "".join(re.findall(r"\d", raw_text))
    return digits or None


def normalize_account_number(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    digits = "".join(re.findall(r"\d", raw_text))
    return digits or None


def normalize_routing_transit_number(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    digits = "".join(re.findall(r"\d", raw_text))
    return digits or None


def normalize_payee_name(raw_text: str | None) -> str | None:
    """Whitespace/formatting cleanup only -- no fuzzy matching or
    substitution (docs/15 S19: normalization should not perform
    aggressive fuzzy matching without carefully defined rules)."""
    if not raw_text:
        return None
    collapsed = re.sub(r"\s+", " ", raw_text).strip()
    return collapsed or None
