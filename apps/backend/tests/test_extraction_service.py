"""Milestone 3 tests: cheque data extraction orchestration
(docs/15_Cheque_Data_Extraction.md).

Uses hand-built OCRRawResult/WordBox fixtures (rather than always running
real Tesseract) so extraction-status/missing-field/no-fabrication logic
can be tested deterministically and quickly. Real end-to-end OCR against
actual Milestone 1 images is covered separately in
test_ocr_milestone1_evaluation.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.extraction.extraction_service import REQUIRED_FIELDS, extract_cheque_data
from app.services.ocr.engine import OCRRawResult, WordBox

IMAGE_W, IMAGE_H = 1200, 560


def _blank_image() -> np.ndarray:
    return np.full((IMAGE_H, IMAGE_W, 3), 250, dtype=np.uint8)


def _word(text, left, top, width=40, height=15, conf=90.0, block=1, par=1, line=1, word=1) -> WordBox:
    return WordBox(text=text, left=left, top=top, width=width, height=height, confidence=conf,
                   block_num=block, par_num=par, line_num=line, word_num=word)


def _complete_words() -> list[WordBox]:
    """Labeled OCR lines placed at arbitrary locations on the image."""
    return [
        _word("DEMO", 30, 30, line=1, word=1),
        _word("Date:", 900, 35, line=2, word=1),
        _word("11/08/2026", 950, 35, line=2, word=2),
        _word("Cheque", 900, 65, line=3, word=1),
        _word("No:", 960, 65, line=3, word=2),
        _word("002020", 990, 65, line=3, word=3),
        _word("Pay", 10, 150, line=4, word=1),
        _word("to", 60, 150, line=4, word=2),
        _word("the", 85, 150, line=4, word=3),
        _word("order", 115, 150, line=4, word=4),
        _word("of:", 160, 150, line=4, word=5),
        _word("Bluepeak", 195, 150, line=4, word=6),
        _word("Distributors", 295, 150, line=4, word=7),
        _word("$17,334.23", 30, 260, width=100, line=5, word=1),
        _word("Seventeen", 30, 330, line=6, word=1),
        _word("Thousand", 120, 330, line=6, word=2),
        _word("Account", 30, 410, line=7, word=1),
        _word("No:", 100, 410, line=7, word=2),
        _word("9000010020", 130, 410, width=100, line=7, word=3),
        _word("Routing/Transit", 30, 430, width=120, line=8, word=1),
        _word("No:", 155, 430, line=8, word=2),
        _word("121000358", 180, 430, width=90, line=8, word=3),
    ]


def _ocr_result(words: list[WordBox], raw_text: str = "", status="COMPLETED", confidence=90.0) -> OCRRawResult:
    return OCRRawResult(
        engine_name="Tesseract", engine_version="5.4.0", raw_text=raw_text, words=words,
        average_confidence=confidence, image_width=IMAGE_W, image_height=IMAGE_H,
        processing_time_ms=10.0, status=status,
    )


class TestCompleteExtraction:
    def test_all_required_fields_extracted(self):
        ocr = _ocr_result(_complete_words())
        result = extract_cheque_data("CHK-TEST-001", ocr, _blank_image())
        assert result.extraction_status == "COMPLETED"
        assert result.missing_fields == []
        assert result.fields["cheque_number"].value == "002020"
        assert result.fields["account_number"].value == "9000010020"
        assert result.fields["routing_transit_number"].value == "121000358"
        assert result.fields["payee_name"].value == "Bluepeak Distributors"
        assert result.fields["amount"].value == 17334.23
        assert result.fields["date"].value == "2026-08-11"

    def test_canonical_schema_shape(self):
        ocr = _ocr_result(_complete_words())
        result = extract_cheque_data("CHK-TEST-002", ocr, _blank_image())
        schema = result.canonical_schema()
        assert set(schema.keys()) == {
            "cheque_id", "cheque_number", "account_number", "routing_transit_number",
            "payee_name", "amount", "amount_in_words", "date", "bank_name", "currency",
            "signature_region_detected", "extraction_status",
        }


class TestMissingFields:
    def test_missing_required_field_yields_partial_status(self):
        words = [w for w in _complete_words() if w.text not in ("Bluepeak", "Distributors")]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-003", ocr, _blank_image())
        assert result.extraction_status == "PARTIAL"
        assert "payee_name" in result.missing_fields

    def test_missing_field_value_is_none_not_fabricated(self):
        words = [w for w in _complete_words() if w.text not in ("Bluepeak", "Distributors")]
        ocr = _ocr_result(words, raw_text="")
        result = extract_cheque_data("CHK-TEST-004", ocr, _blank_image())
        assert result.fields["payee_name"].value is None
        assert result.fields["payee_name"].raw_value is None

    def test_all_fields_missing_when_no_words_at_all(self):
        ocr = _ocr_result([], raw_text="")
        result = extract_cheque_data("CHK-TEST-005", ocr, _blank_image())
        assert result.extraction_status == "PARTIAL"
        assert set(result.missing_fields) == set(REQUIRED_FIELDS)
        for field_name in REQUIRED_FIELDS:
            assert result.fields[field_name].value is None


class TestOCRFailureHandling:
    def test_failed_ocr_produces_failed_extraction_with_no_values(self):
        ocr = _ocr_result([], status="FAILED")
        result = extract_cheque_data("CHK-TEST-006", ocr, _blank_image())
        assert result.extraction_status == "FAILED"
        assert result.fields == {}
        assert set(result.missing_fields) == set(REQUIRED_FIELDS)


class TestNoFabricationBehavior:
    def test_optional_field_left_null_when_absent(self):
        """bank_name/currency are not required fields but must still
        never be guessed when no evidence exists."""
        ocr = _ocr_result([], raw_text="")
        result = extract_cheque_data("CHK-TEST-007", ocr, _blank_image())
        assert result.fields["bank_name"].value is None
        assert result.fields["currency"].value is None

    def test_unparseable_amount_text_is_null_not_a_guessed_number(self):
        words = [_word("Amount:", 30, 260, line=1, word=1), _word("garbled#$%", 90, 260, line=1, word=2)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-008", ocr, _blank_image())
        assert result.fields["amount"].value is None
        # Raw OCR evidence is still preserved even though normalization failed.
        assert result.fields["amount"].raw_value is not None

    def test_unparseable_date_text_is_null_not_a_guessed_date(self):
        words = [_word("Date:", 900, 35, line=1, word=1), _word("garbage", 950, 35, line=1, word=2)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-009", ocr, _blank_image())
        assert result.fields["date"].value is None
        assert result.fields["date"].raw_value is not None


class TestFieldConfidenceHandling:
    def test_region_based_field_reports_measured_confidence(self):
        words = [_word("Pay", 10, 150, line=1, word=1), _word("to", 60, 150, line=1, word=2),
                 _word("Bluepeak", 100, 150, conf=42.5, line=1, word=3),
                 _word("Distributors", 200, 150, conf=88.0, line=1, word=4)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-010", ocr, _blank_image())
        payee = result.fields["payee_name"]
        assert payee.confidence == pytest.approx((42.5 + 88.0) / 2)

    def test_low_confidence_value_is_not_hidden_or_rounded_away(self):
        words = [_word("Pay", 10, 150, line=1, word=1), _word("to", 50, 150, line=1, word=2),
                 _word("Bluepeak", 80, 150, conf=5.0, line=1, word=3)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-011", ocr, _blank_image())
        assert result.fields["payee_name"].confidence == pytest.approx(5.0)
        assert result.fields["payee_name"].value == "Bluepeak"  # still reported, just flagged via confidence

    def test_confidence_is_none_not_a_fake_number_when_unavailable(self):
        """Fallback (full-text keyword) extraction has no specific word
        to attribute confidence to -- it must report None, not a made-up
        percentage."""
        ocr = _ocr_result([], raw_text="Account No: 9000010020")
        result = extract_cheque_data("CHK-TEST-012", ocr, _blank_image())
        assert result.fields["account_number"].value == "9000010020"
        assert result.fields["account_number"].source == "ocr_fallback"
        assert result.fields["account_number"].confidence is None


class TestFallbackKeywordExtraction:
    def test_fallback_used_when_region_has_no_words(self):
        ocr = _ocr_result([], raw_text="Cheque No: 000456\nAccount No: 1234567890")
        result = extract_cheque_data("CHK-TEST-013", ocr, _blank_image())
        assert result.fields["cheque_number"].value == "000456"
        assert result.fields["cheque_number"].source == "ocr_fallback"
        assert result.fields["account_number"].value == "1234567890"

    def test_region_result_preferred_over_fallback_when_both_available(self):
        words = [_word("Cheque", 900, 65, line=1, word=1), _word("No:", 960, 65, line=1, word=2),
                 _word("999999", 990, 65, line=1, word=3)]
        ocr = _ocr_result(words, raw_text="Cheque No: 111111")
        result = extract_cheque_data("CHK-TEST-014", ocr, _blank_image())
        assert result.fields["cheque_number"].value == "999999"
        assert result.fields["cheque_number"].source == "ocr_spatial"


class TestSignatureRegionDetection:
    def test_blank_signature_region_not_detected(self):
        ocr = _ocr_result(_complete_words())
        blank = _blank_image()
        result = extract_cheque_data("CHK-TEST-015", ocr, blank)
        assert result.signature_region_detected is False
        assert result.signature_region_bbox is not None

    def test_signature_region_with_ink_is_detected(self):
        ocr = _ocr_result(_complete_words())
        image = _blank_image()
        # Draw a dense dark scribble inside the signature region
        # (fractions 0.62-0.96 x, 0.70-0.87 y on a 1200x560 canvas).
        image[410:450, 800:1000] = 20
        result = extract_cheque_data("CHK-TEST-016", ocr, image)
        assert result.signature_region_detected is True
        assert result.signature_region_bbox["width"] > 0


class TestCurrencyDetection:
    def test_dollar_sign_detected_as_usd(self):
        words = [_word("$17,334.23", 30, 260, width=100, line=1, word=1)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-017", ocr, _blank_image())
        assert result.fields["currency"].value == "USD"

    def test_no_currency_symbol_leaves_currency_null(self):
        words = [_word("17334.23", 30, 260, width=100, line=1, word=1)]
        ocr = _ocr_result(words)
        result = extract_cheque_data("CHK-TEST-018", ocr, _blank_image())
        assert result.fields["currency"].value is None
