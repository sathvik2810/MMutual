"""Milestone 3 tests: the OCR adapter interface and its Tesseract
implementation (ADR-0002, FR-008).

Exercises TesseractOCREngine directly (not through the API) so the
adapter contract itself -- not just the end-to-end pipeline -- is
verified: given any image, `run()` must return a well-formed
OCRRawResult and must never raise for "just poor recognition" inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.ocr.engine import OCRRawResult
from app.services.ocr.tesseract_engine import TesseractOCREngine, get_ocr_engine, get_tesseract_engine


def _text_image(text: str, size=(600, 200)) -> np.ndarray:
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), text, fill=(0, 0, 0))
    arr = np.array(img)
    return arr[:, :, ::-1].copy()  # RGB -> BGR


class TestAdapterContract:
    def test_engine_singleton_implements_required_properties(self):
        engine = get_ocr_engine()
        assert engine.name == "PaddleOCR"
        assert isinstance(engine.version, str) and engine.version
        assert get_tesseract_engine().name == "Tesseract"

    def test_run_returns_ocr_raw_result(self):
        engine = TesseractOCREngine()
        result = engine.run(_text_image("HELLO WORLD"))
        assert isinstance(result, OCRRawResult)
        assert result.status in ("COMPLETED", "LOW_CONFIDENCE", "PARTIAL", "FAILED")

    def test_run_never_raises_on_blank_image(self):
        blank = np.full((200, 600, 3), 255, dtype=np.uint8)
        engine = TesseractOCREngine()
        result = engine.run(blank)  # must not raise
        assert result.status in ("PARTIAL", "COMPLETED")
        assert result.raw_text.strip() == ""

    def test_run_never_raises_on_random_noise_image(self):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (200, 600, 3), dtype=np.uint8)
        engine = TesseractOCREngine()
        result = engine.run(noise)  # must not raise regardless of garbage input
        assert isinstance(result, OCRRawResult)

    def test_grayscale_image_supported(self):
        gray = np.full((200, 600), 255, dtype=np.uint8)
        engine = TesseractOCREngine()
        result = engine.run(gray)
        assert isinstance(result, OCRRawResult)


class TestRealTextRecognition:
    def test_clear_text_is_recognized(self):
        engine = TesseractOCREngine()
        result = engine.run(_text_image("CHEQUE NUMBER 123456"))
        assert "CHEQUE" in result.raw_text.upper() or "123456" in result.raw_text

    def test_words_have_bounding_boxes_and_confidence(self):
        engine = TesseractOCREngine()
        result = engine.run(_text_image("TESTWORD"))
        assert len(result.words) > 0
        for word in result.words:
            assert word.width > 0
            assert word.height > 0
            assert -1 <= word.confidence <= 100

    def test_processing_time_recorded(self):
        engine = TesseractOCREngine()
        result = engine.run(_text_image("SOME TEXT"))
        assert result.processing_time_ms > 0

    def test_average_confidence_within_valid_range(self):
        engine = TesseractOCREngine()
        result = engine.run(_text_image("READABLE TEXT HERE"))
        assert 0 <= result.average_confidence <= 100
