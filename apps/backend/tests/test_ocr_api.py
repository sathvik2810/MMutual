"""Milestone 3 tests: OCR API endpoints (docs/26_API_Specification.md
S11-S12: POST/GET /cheques/{id}/ocr)."""

from __future__ import annotations

import glob
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.repositories.cheque_repository import get_cheque_repository

client = TestClient(app)

# Absolute, invocation-directory-independent path (works whether pytest is
# run from apps/backend/ or the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_PATHS = glob.glob(str(_REPO_ROOT / "data" / "sample_cheques" / "valid" / "*.png"))


@pytest.fixture(autouse=True)
def _clean_repository():
    yield
    get_cheque_repository().clear_for_testing()


def _upload_real_sample() -> str:
    assert _SAMPLE_PATHS, "Milestone 1 sample cheques not found"
    with open(_SAMPLE_PATHS[0], "rb") as f:
        content = f.read()
    resp = client.post("/api/v1/cheques/upload", files={"file": ("cheque.png", content, "image/png")})
    assert resp.status_code == 201
    return resp.json()["cheque_id"]


def _make_png_bytes(size=(600, 300), color=(250, 249, 244)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestOCRStartEndpoint:
    def test_ocr_runs_successfully_on_real_sample(self):
        cheque_id = _upload_real_sample()
        resp = client.post(f"/api/v1/cheques/{cheque_id}/ocr")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cheque_id"] == cheque_id
        assert body["status"] in ("COMPLETED", "LOW_CONFIDENCE", "PARTIAL")
        assert 0 <= body["ocr_confidence"] <= 100

    def test_ocr_on_unknown_cheque_returns_404(self):
        resp = client.post("/api/v1/cheques/CHK-2026-999999/ocr")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "CHEQUE_NOT_FOUND"

    def test_ocr_before_preprocessing_completes_is_rejected(self):
        """A cheque record that exists but never finished preprocessing
        (simulated directly via the repository) must not be OCR-able."""
        repo = get_cheque_repository()
        repo.save("CHK-2026-000900", {
            "cheque_id": "CHK-2026-000900", "file_name": "x.png", "file_type": "image/png",
            "file_size": 10, "input_source": "UPLOAD", "upload_timestamp": "2026-01-01T00:00:00Z",
            "file_hash": "abc", "original_width": 100, "original_height": 100, "pdf_page_count": None,
            "processing_status": "PROCESSING", "preprocessing": None, "error": None,
        })
        resp = client.post("/api/v1/cheques/CHK-2026-000900/ocr")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "CHEQUE_NOT_PREPROCESSED"


class TestOCRResultEndpoint:
    def test_get_ocr_result_after_running_ocr(self):
        cheque_id = _upload_real_sample()
        client.post(f"/api/v1/cheques/{cheque_id}/ocr")
        resp = client.get(f"/api/v1/cheques/{cheque_id}/ocr")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cheque_id"] == cheque_id
        assert body["engine"] in {"PaddleOCR", "Tesseract"}
        assert "extracted_data" in body
        assert body["engine"] in {"PaddleOCR", "Tesseract"}
        assert "words" in body and "extraction_warnings" in body and "validation_results" in body
        assert set(body["extracted_data"].keys()) >= {
            "cheque_number", "account_number", "routing_transit_number",
            "payee_name", "amount", "amount_in_words", "date", "bank_name", "currency",
        }

    def test_get_ocr_result_before_running_ocr_returns_404(self):
        cheque_id = _upload_real_sample()
        resp = client.get(f"/api/v1/cheques/{cheque_id}/ocr")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "OCR_NOT_RUN"

    def test_get_ocr_result_for_unknown_cheque_returns_404(self):
        resp = client.get("/api/v1/cheques/CHK-2026-999999/ocr")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "CHEQUE_NOT_FOUND"

    def test_extracted_field_shape_matches_documented_metadata_model(self):
        cheque_id = _upload_real_sample()
        client.post(f"/api/v1/cheques/{cheque_id}/ocr")
        body = client.get(f"/api/v1/cheques/{cheque_id}/ocr").json()
        cheque_number_field = body["extracted_data"]["cheque_number"]
        assert set(cheque_number_field.keys()) == {"value", "raw_value", "confidence", "source", "validation_status"}
        # Milestone 4 (Validation Engine) hasn't run yet -- must not be
        # fabricated as pass/fail this early.
        assert cheque_number_field["validation_status"] is None


class TestFullPipelineViaRealMilestone1Data:
    def test_upload_preprocess_ocr_extract_end_to_end(self):
        cheque_id = _upload_real_sample()
        detail_before_ocr = client.get(f"/api/v1/cheques/{cheque_id}").json()
        assert detail_before_ocr["preprocessing"]["preprocessing_status"] == "COMPLETED"

        ocr_start = client.post(f"/api/v1/cheques/{cheque_id}/ocr")
        assert ocr_start.status_code == 200

        result = client.get(f"/api/v1/cheques/{cheque_id}/ocr").json()
        assert result["extraction_status"] == "COMPLETED"
        assert result["extracted_data"]["cheque_number"]["value"] is not None
        assert result["extracted_data"]["amount"]["value"] is not None
