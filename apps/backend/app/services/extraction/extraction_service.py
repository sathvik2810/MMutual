"""Cheque data extraction orchestrator (docs/15_Cheque_Data_Extraction.md).

Converts an OCRRawResult (raw text + word bounding boxes) into the
canonical structured cheque record using semantic labels, reading order,
OCR confidence, and field-pattern validation. Cheque-specific coordinates
and templates are not used for cheque data extraction.

Hard rule enforced throughout: a field that cannot be reliably read is
left null with its raw OCR evidence (if any) preserved -- never a
fabricated or guessed value (docs/15 S32).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services.extraction import field_parsers, normalization
from app.services.ocr.engine import OCRRawResult, WordBox
from app.services.ocr.regions import REGIONS

REQUIRED_FIELDS = ("cheque_number", "account_number", "payee_name", "amount", "date")

_NORMALIZERS = {
    "cheque_number": normalization.normalize_cheque_number,
    "account_number": normalization.normalize_account_number,
    "routing_transit_number": normalization.normalize_routing_transit_number,
    "payee_name": normalization.normalize_payee_name,
    "amount": normalization.normalize_amount,
    "date": normalization.normalize_date,
}


@dataclass
class FieldExtraction:
    value: Any
    raw_value: str | None
    confidence: float | None
    source: str | None  # "ocr_region" | "ocr_fallback" | None
    validation_status: str | None = None  # not populated until Milestone 4

    def as_dict(self) -> dict:
        reliability = "NOT_DETECTED" if self.value in (None, "") else (
            "UNCERTAIN" if self.validation_status == "UNCERTAIN" or self.confidence is None
            or self.confidence < settings.ocr_field_low_confidence_threshold else "RELIABLE"
        )
        return {
            "value": self.value,
            "raw_value": self.raw_value,
            "confidence": round(self.confidence, 2) if self.confidence is not None else None,
            "source": self.source,
            "validation_status": self.validation_status,
            "reliability": reliability,
        }


@dataclass
class ChequeExtractionResult:
    cheque_id: str
    template: str
    fields: dict[str, FieldExtraction] = dataclass_field(default_factory=dict)
    signature_region_detected: bool = False
    signature_region_bbox: dict | None = None
    extraction_status: str = "PENDING"
    missing_fields: list[str] = dataclass_field(default_factory=list)
    ambiguous_fields: list[str] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    validation_results: dict[str, Any] = dataclass_field(default_factory=dict)
    processing_time_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "cheque_id": self.cheque_id,
            "template": self.template,
            "extraction_status": self.extraction_status,
            "missing_fields": self.missing_fields,
            "ambiguous_fields": self.ambiguous_fields,
            "warnings": self.warnings,
            "validation_results": self.validation_results,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "signature_region_detected": self.signature_region_detected,
            "signature_region_bbox": self.signature_region_bbox,
            "fields": {name: f.as_dict() for name, f in self.fields.items()},
        }

    def canonical_schema(self) -> dict:
        """docs/15_Cheque_Data_Extraction.md S29 canonical output shape."""
        return {
            "cheque_id": self.cheque_id,
            "cheque_number": self.fields["cheque_number"].value,
            "account_number": self.fields["account_number"].value,
            "routing_transit_number": self.fields["routing_transit_number"].value,
            "payee_name": self.fields["payee_name"].value,
            "amount": self.fields["amount"].value,
            "amount_in_words": self.fields["amount_in_words"].value,
            "date": self.fields["date"].value,
            "bank_name": self.fields["bank_name"].value,
            "currency": self.fields["currency"].value,
            "signature_region_detected": self.signature_region_detected,
            "extraction_status": self.extraction_status,
        }


def _detect_signature_region(image: np.ndarray, image_width: int, image_height: int) -> tuple[bool, dict]:
    """Detects whether the signature region contains meaningful ink
    content -- region *detection* only, no comparison/verification
    (that's Milestone 6's job)."""
    region = REGIONS["signature"]
    x0, y0 = int(region.x0 * image_width), int(region.y0 * image_height)
    x1, y1 = int(region.x1 * image_width), int(region.y1 * image_height)
    bbox = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return False, bbox

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(np.count_nonzero(thresh)) / thresh.size
    detected = ink_ratio >= settings.signature_region_ink_ratio_threshold
    return detected, bbox


def _row_neighbours(label: WordBox, words: list[WordBox], *, right_only: bool = True) -> list[WordBox]:
    """Find OCR boxes on the label's visual row using box height as scale."""
    result = []
    for word in words:
        if word is label or (right_only and word.left < label.right):
            continue
        vertical_overlap = min(label.bottom, word.bottom) - max(label.top, word.top)
        center_delta = abs(label.center[1] - word.center[1])
        if vertical_overlap > 0 or center_delta <= max(label.height, word.height) * 0.6:
            result.append(word)
    return sorted(result, key=lambda word: word.left)


def _join_overlapping_numeric_boxes(boxes: list[WordBox]) -> str:
    """Join date fragments, removing a duplicated digit only when OCR boxes overlap."""
    fragments = [re.sub(r"\D", "", word.text) for word in boxes]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return ""
    value = fragments[0]
    for previous, word, fragment in zip(boxes, boxes[1:], fragments[1:]):
        overlap = previous.right - word.left
        if overlap > 0:
            duplicate = 0
            for size in range(1, min(len(value), len(fragment), 3) + 1):
                if value.endswith(fragment[:size]):
                    duplicate = size
            value += fragment[duplicate:]
        else:
            value += fragment
    return value


def extract_cheque_data(cheque_id: str, ocr_result: OCRRawResult, image: np.ndarray) -> ChequeExtractionResult:
    start = time.perf_counter()

    if ocr_result.status == "FAILED":
        return ChequeExtractionResult(
            cheque_id=cheque_id, template="GENERALIZED_LAYOUT", extraction_status="FAILED",
            missing_fields=list(REQUIRED_FIELDS), processing_time_ms=(time.perf_counter() - start) * 1000,
        )

    words = ocr_result.words
    raw_text = ocr_result.raw_text
    width, height = ocr_result.image_width, ocr_result.image_height

    # Build reading-order lines from OCR metadata. No cheque template or
    # absolute/fractional field coordinates are used for data extraction.
    grouped: dict[tuple[int, int, int], list[WordBox]] = {}
    for word in words:
        grouped.setdefault((word.block_num, word.par_num, word.line_num), []).append(word)
    lines = [" ".join(w.text for w in sorted(row, key=lambda item: (item.left, item.word_num)))
             for _, row in sorted(grouped.items())]
    # Prefer the OCR tokens tied to positions when available; raw full-text
    # remains the fallback when the engine cannot provide boxes.
    text = "\n".join(lines) if lines else raw_text.strip()

    def field(name: str, candidate: str | None, pattern_name: str | None = None) -> FieldExtraction:
        if not candidate and name != "payee_name":
            candidate = field_parsers.parse_field_from_text(pattern_name or name, text)
        if not candidate:
            return FieldExtraction(None, None, None, None)
        candidate = field_parsers.strip_known_label(pattern_name or name, candidate)
        normalized = _NORMALIZERS[name](candidate) if name in _NORMALIZERS else candidate
        candidate_lower = candidate.lower()
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_lower))
        evidence_words = [w for w in words if candidate_tokens.intersection(
            re.findall(r"[a-z0-9]+", w.text.lower())) or (
                len(re.sub(r"\D", "", w.text)) >= 3
                and re.sub(r"\D", "", w.text) in re.sub(r"\D", "", candidate))]
        confidence = (sum(w.confidence for w in evidence_words) / len(evidence_words)) if evidence_words else None
        return FieldExtraction(normalized, candidate, confidence, "ocr_spatial" if confidence is not None else "ocr_fallback")

    def labelled(pattern: str) -> str | None:
        match = re.search(pattern, text, re.I)
        return match.group(1).strip() if match else None

    cheque_no = field("cheque_number", labelled(r"(?:Cheque|Check|Chq)\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9]{4,12})"))
    account = field("account_number", labelled(r"(?:Account|A\s*/?\s*C)\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9]{6,24})"))
    routing = field("routing_transit_number", labelled(r"(?:Routing\s*/?\s*Transit|Routing)\s*(?:No\.?|Code)?\s*[:\-]?\s*([A-Za-z0-9]{6,16})"))
    ifsc_match = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", text, re.I)
    ifsc_words = [w for w in words if ifsc_match and ifsc_match.group(0).lower() in w.text.lower()]
    ifsc_confidence = sum(w.confidence for w in ifsc_words) / len(ifsc_words) if ifsc_words else None
    ifsc = FieldExtraction(ifsc_match.group(0).upper(), ifsc_match.group(0), ifsc_confidence, "ocr_pattern") if ifsc_match else FieldExtraction(None, None, None, None)
    date_label = next((w for w in words if re.fullmatch(r"date|dated", w.text.strip(".:"), re.I)), None)
    date_candidate = None
    if date_label:
        date_fragments = [w for w in _row_neighbours(date_label, words)
                          if re.search(r"\d", w.text) and re.fullmatch(r"[\d/.,-]+", w.text.strip())]
        if date_fragments:
            date_candidate = _join_overlapping_numeric_boxes(date_fragments)
    if date_candidate is None:
        date_candidate = labelled(r"(?:Date|Dated)\s*[:\-]?\s*(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{1,2}[-/ ]+[A-Za-z]{3,9}[-/ ]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{8})")
    if date_candidate is None:
        labelled_date_text = labelled(r"(?:Date|Dated)\s*[:\-]?\s*([^\n]+)")
        if labelled_date_text and normalization.normalize_date(labelled_date_text) is None:
            date_candidate = labelled_date_text
        else:
            date_candidate = field_parsers.parse_unlabelled_date(text)
    date = field("date", date_candidate)
    payee_label = next((w for w in words if re.fullmatch(r"pay(?:ee)?", w.text.strip(".:"), re.I)), None)
    payee_value = None
    if payee_label:
        payee_boxes = [w for w in _row_neighbours(payee_label, words)
                       if re.search(r"[A-Za-z]", w.text)
                       and w.text.strip(".:,-").lower() not in {"to", "the", "order", "of"}]
        if payee_boxes:
            payee_value = " ".join(w.text for w in payee_boxes)
    if payee_value is None:
        payee_value = labelled(r"(?:Pay\s+to(?:\s+the\s+order\s+of)?|Payee)\s*[:\-]?\s*([^\n]+)")
    if payee_value is None:
        # A label may be one OCR line and the handwritten value the next.
        for i, line in enumerate(lines[:-1]):
            if re.search(r"\b(?:Pay|Payee)\b", line, re.I):
                payee_value = lines[i + 1]
                break
    if payee_value:
        # Negotiable-instrument boilerplate often follows the handwritten name.
        payee_value = re.sub(r"\s+or\s*bearer\b.*$", "", payee_value, flags=re.I).strip()
    if payee_value and (not re.fullmatch(r"[A-Za-z][A-Za-z .,&'-]*", payee_value.strip())
                        or re.search(r"\b(?:date|amount|account|cheque|check|bank|rupees|ifsc)\b", payee_value, re.I)):
        payee_value = None
    payee = field("payee_name", payee_value)

    amount_match = re.search(r"(?:Amount\s*[:\-]?\s*|[$]|Rs\.?\s*|INR\s*)([\d,]+(?:\.\d{1,2})?)", text, re.I)
    amount_raw = amount_match.group(1) if amount_match else None
    if amount_raw is None:
        amount_label_text = labelled(r"Amount(?!\s+in\s+words)\s*[:\-]?\s*([^\n]+)")
        amount_raw = amount_label_text.strip() if amount_label_text else None
    if amount_raw is None:
        # A printed/handwritten RUPEES label and a numeric box on the same
        # visual row provide currency context even when no symbol is present.
        rupees_label = next((w for w in words if re.fullmatch(r"rupees?", w.text.strip(".:"), re.I)), None)
        if rupees_label:
            numeric_boxes = [w for w in _row_neighbours(rupees_label, words)
                             if re.fullmatch(r"(?:\d{1,3}(?:,\d{2,3})+|\d{3,})(?:\.\d{1,2})?", w.text.strip())]
            if numeric_boxes:
                amount_raw = max(numeric_boxes, key=lambda w: (w.left, w.confidence)).text
        if amount_raw is None:
            # Without a label or currency marker accept only conventional
            # grouped/decimal forms; plain identifiers are not money evidence.
            amount_match = re.search(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b|\b\d+\.\d{1,2}\b", text)
            amount_raw = amount_match.group(0) if amount_match else None
    amount = field("amount", amount_raw)
    words_match = re.search(r"(?:Amount\s+in\s+words|Rupees)\s*[:\-]?\s*([^\n]+)", text, re.I)
    amount_words_value = words_match.group(1) if words_match else None
    rupees_label = next((w for w in words if re.fullmatch(r"rupees?", w.text.strip(".:"), re.I)), None)
    if rupees_label:
        phrase_boxes = [w for w in _row_neighbours(rupees_label, words)
                        if re.search(r"[A-Za-z]", w.text)
                        and not re.search(r"\b(?:only|payee|account|date|ifsc)\b", w.text, re.I)]
        if phrase_boxes:
            amount_words_value = " ".join(w.text for w in phrase_boxes)
    if amount_words_value is None:
        from app.services.validation.amount_words import words_to_amount
        phrases = [line for line in lines if len(line.split()) >= 3 and words_to_amount(line) is not None]
        amount_words_value = max(phrases, key=len) if phrases else None
    amount_words = field("amount_in_words", amount_words_value) if amount_words_value else FieldExtraction(None, None, None, None)
    bank_line = next((line for line in lines if re.search(r"\bbank\b", line, re.I)), None)
    bank = FieldExtraction(bank_line, bank_line, None, "ocr_fallback") if bank_line else FieldExtraction(None, None, None, None)
    fields = {"cheque_number": cheque_no, "account_number": account, "routing_transit_number": routing,
              "payee_name": payee, "date": date, "amount": amount, "amount_in_words": amount_words,
              "ifsc": ifsc, "micr": FieldExtraction(None, None, None, None), "bank_name": bank,
              "currency": FieldExtraction("INR", "Rs/INR", None, "ocr_text") if re.search(r"(?:₹|\bRs\.?\b|\bINR\b|\bRupees\b)", text, re.I)
              else FieldExtraction("USD", "$", None, "ocr_text") if "$" in text else FieldExtraction(None, None, None, None)}
    micr_words = [w for w in words if height and w.center[1] >= height * 0.78 and re.search(r"\d{6,}", w.text)]
    if micr_words:
        micr_raw = " ".join(w.text for w in sorted(micr_words, key=lambda item: item.left))
        fields["micr"] = FieldExtraction(re.sub(r"\D", "", micr_raw), micr_raw,
                                             sum(w.confidence for w in micr_words)/len(micr_words), "ocr_spatial")
        if fields["cheque_number"].value is None:
            # In Indian MICR encoding the leading six digits carry the
            # cheque serial number. Require a full MICR-length digit sequence
            # so an arbitrary footer number is not promoted to an identifier.
            micr_digits = fields["micr"].value
            if re.fullmatch(r"\d{15,}", micr_digits or ""):
                fields["cheque_number"] = FieldExtraction(
                    micr_digits[:6], micr_raw, fields["micr"].confidence, "ocr_micr_pattern")

    signature_detected, signature_bbox = _detect_signature_region(image, width, height)

    missing_fields = [name for name in REQUIRED_FIELDS if fields[name].value in (None, "")]
    ambiguous_fields: list[str] = []  # no multi-candidate disambiguation implemented in this milestone

    if missing_fields:
        extraction_status = "PARTIAL"
    else:
        extraction_status = "COMPLETED"

    warnings = [f"{name} was not detected reliably." for name in missing_fields]
    threshold = settings.ocr_field_low_confidence_threshold
    warnings.extend(f"{name} has low OCR confidence." for name, value in fields.items()
                    if value.value is not None and value.confidence is not None and value.confidence < threshold)
    numeric_amount = fields["amount"].value
    words_amount = None
    if fields["amount_in_words"].value:
        from app.services.validation.amount_words import words_to_amount
        words_amount = words_to_amount(fields["amount_in_words"].value)
    amount_status = "NOT_CHECKED"
    if numeric_amount is not None and words_amount is not None:
        amount_status = "PASS" if abs(float(numeric_amount) - words_amount) < 0.01 else "FAIL"
    elif fields["amount_in_words"].value and words_amount is None:
        amount_status = "UNCERTAIN"
        fields["amount_in_words"].validation_status = "UNCERTAIN"
        warnings.append("amount_in_words could not be validated from the recognized wording.")
    if amount_status == "FAIL":
        warnings.append("Numeric amount and amount in words do not match.")
    return ChequeExtractionResult(
        cheque_id=cheque_id,
        template="GENERALIZED_LAYOUT",
        fields=fields,
        signature_region_detected=signature_detected,
        signature_region_bbox=signature_bbox,
        extraction_status=extraction_status,
        missing_fields=missing_fields,
        ambiguous_fields=ambiguous_fields,
        warnings=warnings,
        validation_results={"amount_consistency": {"status": amount_status,
            "numeric_amount": numeric_amount, "words_amount": words_amount}},
        processing_time_ms=(time.perf_counter() - start) * 1000,
    )
