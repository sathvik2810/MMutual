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
