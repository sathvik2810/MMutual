"""Tesseract OCR adapter implementation (ADR-0002).

The only module in the codebase allowed to import pytesseract. Everything
else talks to the `OCREngine` protocol from engine.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytesseract
from PIL import Image

from app.core.config import settings
from app.services.ocr.engine import OCRRawResult, WordBox
from app.services.ocr.exceptions import OCREngineUnavailableError

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd_path

_DEFAULT_CONFIG = "--oem 3 --psm 6"


class TesseractOCREngine:
    """OCREngine implementation backed by Tesseract via PyTesseract."""

    def __init__(self) -> None:
        self._version: str | None = None

    @property
    def name(self) -> str:
        return "Tesseract"

    @property
    def version(self) -> str:
        if self._version is None:
            try:
                self._version = str(pytesseract.get_tesseract_version())
            except Exception:  # noqa: BLE001
                self._version = "unknown"
        return self._version

    def run(self, image: np.ndarray, *, config: str | None = None) -> OCRRawResult:
        start = time.perf_counter()
        pil_image = self._to_pil(image)
        height, width = pil_image.height, pil_image.width
        run_config = config or _DEFAULT_CONFIG

        try:
            data = pytesseract.image_to_data(
                pil_image, config=run_config, output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCREngineUnavailableError(
                "Tesseract binary not found. Check settings.tesseract_cmd_path."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            return OCRRawResult(
                engine_name=self.name,
                engine_version=self.version,
                raw_text="",
                words=[],
                average_confidence=0.0,
                image_width=width,
                image_height=height,
                processing_time_ms=elapsed_ms,
                status="FAILED",
                error_message=str(exc),
            )

        words: list[WordBox] = []
        confidences: list[float] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            words.append(WordBox(
                text=text,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=conf,
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
                line_num=int(data["line_num"][i]),
                word_num=int(data["word_num"][i]),
            ))
            if conf >= 0:
                confidences.append(conf)

        raw_text = pytesseract.image_to_string(pil_image, config=run_config)
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not words:
            status = "PARTIAL"
        elif average_confidence < settings.ocr_low_confidence_threshold:
            status = "LOW_CONFIDENCE"
        else:
            status = "COMPLETED"

        return OCRRawResult(
            engine_name=self.name,
            engine_version=self.version,
            raw_text=raw_text,
            words=words,
            average_confidence=average_confidence,
            image_width=width,
            image_height=height,
            processing_time_ms=elapsed_ms,
            status=status,
        )

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            return Image.fromarray(image)
        # Assume BGR (OpenCV convention) -> convert to RGB for Pillow/Tesseract.
        return Image.fromarray(image[:, :, ::-1])


_engine_instance: TesseractOCREngine | None = None


def get_tesseract_engine() -> TesseractOCREngine:
    """Return the Tesseract fallback adapter."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TesseractOCREngine()
    return _engine_instance


def get_ocr_engine():
    """Return the configured primary engine (PaddleOCR)."""
    from app.services.ocr.paddle_engine import get_paddle_engine
    return get_paddle_engine()
