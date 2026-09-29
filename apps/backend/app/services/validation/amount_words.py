"""Parses an English amount-in-words string back into a number, for the
AMOUNT_CONSISTENCY check (docs/16_Validation_Engine.md S22).

This is the inverse of the word-generation logic in
scripts/generate_synthetic_data.py's amount_to_words(), so it correctly
interprets the Milestone 1 dataset's own cheque images (e.g. "Seventeen
Thousand Three Hundred Thirty Four and 23/100 Only" -> 17334.23). If any
word in the text isn't recognized, parsing stops and returns None rather
than guessing at a partial amount (no-fabrication principle, consistent
with Milestone 3's normalization functions).

The cents fraction ("NN/100") is digits, not letters, so it is matched
separately with its own regex before the letters-only word tokenizer
runs -- otherwise it would be silently dropped rather than parsed.
"""

from __future__ import annotations

import re

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
           "lac": 100_000, "lacs": 100_000, "lack": 100_000, "lacks": 100_000,
           "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000}
_IGNORED_WORDS = {"only", "and", "rupees", "dollars", "rs"}
_CENTS_FRACTION_RE = re.compile(r"(\d{1,2})\s*/\s*100")


def words_to_amount(text: str | None) -> float | None:
    if not text:
        return None

    cents = 0
    cents_match = _CENTS_FRACTION_RE.search(text)
    if cents_match:
        cents = int(cents_match.group(1))
        text = text[: cents_match.start()] + text[cents_match.end():]

    words = [w for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in _IGNORED_WORDS]
    if not words:
        return None

    total = 0
    current = 0
    recognized_any = False

    for word in words:
        if word in _ONES:
            current += _ONES[word]
            recognized_any = True
        elif word in _TENS:
            current += _TENS[word]
            recognized_any = True
        elif word == "hundred":
            current = (current or 1) * 100
            recognized_any = True
        elif word in ("thousand", "million", "lakh", "lakhs", "lac", "lacs", "lack", "lacks", "crore", "crores"):
            total += (current or 1) * _SCALES[word]
            current = 0
            recognized_any = True
        else:
            return None  # unrecognized word -- do not guess the rest of the amount

    if not recognized_any:
        return None
    return float(total + current) + (cents / 100.0)
