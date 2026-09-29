"""Primary/fallback OCR and layout independent extraction contracts."""
from pathlib import Path

import numpy as np
import pytest

from app.services.extraction.extraction_service import extract_cheque_data
from app.services.ocr.engine import OCRRawResult
from app.services.ocr.ocr_service import run_ocr_for_cheque
from app.services.validation.amount_words import words_to_amount


class _Paddle:
    name, version = "PaddleOCR", "fake"
    def __init__(self, result=None, error=None): self.result, self.error = result, error
    def run(self, image, **kwargs):
        if self.error: raise RuntimeError("model unavailable")
        return self.result


class _Tesseract:
    name, version = "Tesseract", "fake"
    def __init__(self): self.called = False
    def run(self, image, **kwargs):
        self.called = True
        return OCRRawResult(self.name, self.version, "Payee: Mira Traders", [], 88, 800, 400, status="COMPLETED")


def test_paddle_result_keeps_text_confidence_and_boxes():
    from app.services.ocr.paddle_engine import PaddleOCREngine
    class Fake:
        def predict(self, image):
            return [{"dt_polys": [np.array([[1, 2], [41, 2], [41, 20], [1, 20]])],
                     "rec_texts": ["Payee"], "rec_scores": [0.93]}]
    result = PaddleOCREngine(Fake()).run(np.zeros((40, 80, 3), dtype=np.uint8))
    assert result.raw_text == "Payee"
    assert result.words[0].confidence == pytest.approx(93)
    assert (result.words[0].left, result.words[0].top, result.words[0].width) == (1, 2, 40)


def test_paddle_failure_automatically_uses_tesseract(monkeypatch):
    fallback = _Tesseract()
    monkeypatch.setattr("app.services.ocr.ocr_service.load_image_bgr", lambda _: np.zeros((20, 20, 3)))
    monkeypatch.setattr("app.services.ocr.ocr_service.get_tesseract_engine", lambda: fallback)
    outcome = run_ocr_for_cheque("processed", "original", engine=_Paddle(error=True))
    assert fallback.called
    assert outcome.result.engine_name == "Tesseract"
    assert "model unavailable" in outcome.result.error_message


def test_insufficient_paddle_text_automatically_uses_tesseract(monkeypatch):
    fallback = _Tesseract()
    primary = OCRRawResult("PaddleOCR", "fake", "one", [], 91, 20, 20, status="COMPLETED")
    monkeypatch.setattr("app.services.ocr.ocr_service.load_image_bgr", lambda _: np.zeros((20, 20, 3)))
    monkeypatch.setattr("app.services.ocr.ocr_service.get_tesseract_engine", lambda: fallback)
    outcome = run_ocr_for_cheque("processed", "original", engine=_Paddle(result=primary))
    assert fallback.called
    assert outcome.result.engine_name == "Tesseract"


def test_tesseract_replaces_usable_but_incomplete_paddle_extraction(monkeypatch):
    from app.services.ocr.engine import WordBox

    primary_result = OCRRawResult(
        "PaddleOCR", "fake", "Account No 1234567890", [
            WordBox("Account", 1, 1, 30, 10, 95), WordBox("1234567890", 35, 1, 80, 10, 95),
        ], 95, 200, 100, status="COMPLETED",
    )
    fallback_result = OCRRawResult(
        "Tesseract", "fake", "Payee Mira Traders", [
            WordBox("Payee", 1, 1, 30, 10, 90), WordBox("Mira", 35, 1, 30, 10, 92),
        ], 91, 200, 100, status="COMPLETED",
    )

    class Fallback:
        name, version = "Tesseract", "fake"
        def run(self, image, **kwargs): return fallback_result

    monkeypatch.setattr("app.services.ocr.ocr_service.load_image_bgr", lambda _: np.zeros((20, 20, 3)))
    monkeypatch.setattr("app.services.ocr.ocr_service.get_tesseract_engine", lambda: Fallback())
    monkeypatch.setattr(
        "app.services.ocr.ocr_service._extraction_quality",
        lambda result, image: 4 if result.engine_name == "PaddleOCR" else 6,
    )
    outcome = run_ocr_for_cheque("processed", "original", engine=_Paddle(result=primary_result))

    assert outcome.result.engine_name == "Tesseract"
    assert outcome.attempts == 2
    assert "stronger field evidence" in outcome.result.error_message


def test_labeled_extraction_supports_different_text_layout_and_formats():
    from app.services.ocr.engine import OCRRawResult
    text = """Account No: 9876543210
Date: 2026-09-28
Pay to the order of: Mira Traders
Amount: Rs. 1,500/-
Amount in words: One Thousand Five Hundred Rupees Only
IFSC: ABCD0123456
Cheque No: 123456"""
    ocr = OCRRawResult("PaddleOCR", "fake", text, [], 87, 1200, 600, status="COMPLETED")
    result = extract_cheque_data("CHK-LAYOUT", ocr, np.full((600, 1200, 3), 255, dtype=np.uint8))
    assert result.fields["date"].value == "2026-09-28"
    assert result.fields["amount"].value == 1500
    assert result.fields["ifsc"].value == "ABCD0123456"


def test_unlabelled_date_is_extracted_when_unique_valid_candidate_exists():
    text = "Payee: Mira Traders\n01-Oct-2026\nAmount: Rs. 1,500/-"
    ocr = OCRRawResult("PaddleOCR", "fake", text, [], 87, 1200, 600, status="COMPLETED")
    result = extract_cheque_data("CHK-UNLABELLED-DATE", ocr, np.full((600, 1200, 3), 255, dtype=np.uint8))
    assert result.fields["date"].value == "2026-10-01"


def test_unlabelled_date_is_not_guessed_when_multiple_candidates_exist():
    text = "01-Oct-2026\n02-Oct-2026\nAmount: 1,000\nAmount in words: One Thousand Rupees Only"
    ocr = OCRRawResult("PaddleOCR", "fake", text, [], 87, 1200, 600, status="COMPLETED")
    result = extract_cheque_data("CHK-AMBIGUOUS-DATE", ocr, np.full((600, 1200, 3), 255, dtype=np.uint8))
    assert result.fields["date"].value is None
    assert result.fields["micr"].value is None
    assert result.validation_results["amount_consistency"]["status"] == "PASS"


def test_indian_lakh_wording_is_parsed_without_guessing():
    assert words_to_amount("One Lakh Twenty Five Thousand Rupees Only") == 125000
    assert words_to_amount("One Thousand Nonsense Rupees") is None


def test_spatial_labels_extract_compact_date_payee_and_amount_with_boxes():
    from app.services.ocr.engine import WordBox

    def box(text, x, y, w=80, h=24, conf=94, line=1):
        return WordBox(text, x, y, w, h, conf, 1, 1, line, 1)

    words = [
        box("Date", 1000, 40, 70, 24),
        # Adjacent overlapping recognitions are fragments of one labeled date.
        box("25012", 1090, 40, 110, 32, line=2),
        box("2016", 1190, 42, 100, 30, line=3),
        box("Pay", 20, 120, 55, 28, line=4),
        box("Mira Traders", 130, 115, 250, 38, 91, line=5),
        box("ORBEARER/", 410, 118, 140, 30, 90, line=5),
        box("RUPEES", 20, 220, 100, 28, line=6),
        box("One Lakh Twenty Five Thousand", 150, 215, 430, 38, 93, line=7),
        box("125000", 730, 220, 160, 34, 96, line=8),
        box("30913550021101242616031", 100, 520, 500, 20, 97, line=9),
    ]
    ocr = OCRRawResult("PaddleOCR", "fake", "", words, 94, 1200, 560, status="COMPLETED")
    result = extract_cheque_data("CHK-SPATIAL", ocr, np.full((560, 1200, 3), 255, dtype=np.uint8))

    assert result.fields["date"].value == "2016-01-25"
    assert result.fields["date"].confidence == pytest.approx(94)
    assert result.fields["payee_name"].value == "Mira Traders"
    assert result.fields["payee_name"].confidence == pytest.approx(91)
    assert result.fields["amount"].value == 125000
    assert result.fields["amount"].confidence == pytest.approx(96)
    assert result.fields["amount_in_words"].value == "One Lakh Twenty Five Thousand"
    assert result.validation_results["amount_consistency"]["status"] == "PASS"
    assert result.fields["micr"].value == "30913550021101242616031"
    assert result.fields["cheque_number"].value == "309135"


def test_unparseable_amount_words_remain_available_but_are_marked_uncertain():
    from app.services.ocr.engine import WordBox
    words = [
        WordBox("RUPEES", 10, 100, 80, 20, 98, 1, 1, 1, 1),
        WordBox("One Lalch Twenty Five Thousand", 120, 95, 350, 30, 98, 1, 1, 1, 2),
        WordBox("125000", 600, 100, 120, 30, 99, 1, 1, 1, 3),
    ]
    ocr = OCRRawResult("PaddleOCR", "fake", "", words, 98, 800, 500, status="COMPLETED")
    result = extract_cheque_data("CHK-UNCERTAIN-WORDS", ocr, np.full((500, 800, 3), 255, dtype=np.uint8))
    amount_words = result.fields["amount_in_words"].as_dict()
    assert amount_words["value"] == "One Lalch Twenty Five Thousand"
    assert amount_words["reliability"] == "UNCERTAIN"
    assert result.validation_results["amount_consistency"]["status"] == "UNCERTAIN"


def test_handwritten_axis_sample_runs_through_ocr_and_fraud_pipeline_when_available():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.repositories.cheque_repository import get_cheque_repository
    sample = Path(__file__).resolve().parents[3] / "data/IDRBT/300/Cheque 309135.tif"
    if not sample.exists():
        pytest.skip("Optional handwritten Axis Bank sample is not present in this checkout.")
    client = TestClient(app)
    try:
        response = client.post("/api/v1/cheques/upload", files={"file": (sample.name, sample.read_bytes(), "image/tiff")})
        assert response.status_code == 201
        cheque_id = response.json()["cheque_id"]
        assert client.post(f"/api/v1/cheques/{cheque_id}/ocr").status_code == 200
        ocr = client.get(f"/api/v1/cheques/{cheque_id}/ocr").json()
        assert ocr["engine"] in {"PaddleOCR", "Tesseract"}
        assert ocr["raw_text"]
        assert "payee_name" in ocr["extracted_data"]
        assert client.post(f"/api/v1/cheques/{cheque_id}/validate").status_code == 200
        assert client.post(f"/api/v1/cheques/{cheque_id}/fraud-analysis").status_code == 200
    finally:
        get_cheque_repository().clear_for_testing()
