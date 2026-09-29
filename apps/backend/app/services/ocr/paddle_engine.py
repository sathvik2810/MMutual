"""PaddleOCR adapter. Paddle is loaded lazily so Tesseract remains usable
when its optional runtime or model download is unavailable."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from app.core.config import settings
from app.services.ocr.engine import OCRRawResult, WordBox
from app.services.ocr.exceptions import OCREngineUnavailableError


class PaddleOCREngine:
    def __init__(self, ocr=None):
        self._ocr = ocr

    @property
    def name(self) -> str:
        return "PaddleOCR"

    @property
    def version(self) -> str:
        try:
            import paddleocr
            return getattr(paddleocr, "__version__", "unknown")
        except Exception:
            return "unknown"

    def _instance(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang="en")
            except Exception as exc:
                raise OCREngineUnavailableError(f"PaddleOCR could not be initialized: {exc}") from exc
        return self._ocr

    def run(self, image: np.ndarray, *, config: str | None = None) -> OCRRawResult:
        start = time.perf_counter()
        height, width = image.shape[:2]
        try:
            engine = self._instance()
            if hasattr(engine, "predict"):
                results = engine.predict(image)
            else:  # PaddleOCR 2.x compatibility
                results = engine.ocr(image, cls=True)
            words: list[WordBox] = []
            lines: list[str] = []
            confidences: list[float] = []
            for page in results or []:
                if hasattr(page, "json"):
                    page_json = page.json
                    page = page_json() if callable(page_json) else page_json
                if isinstance(page, dict) and "res" in page:
                    page = page["res"]
                if isinstance(page, dict) or hasattr(page, "__getitem__") and hasattr(page, "keys"):
                    page = dict(page)
                    polygons = page.get("dt_polys", page.get("rec_polys", []))
                    texts, scores = page.get("rec_texts", []), page.get("rec_scores", [])
                    rows = [(poly, (text, score)) for poly, text, score in zip(polygons, texts, scores)]
                else:
                    rows = page or []
                for row in rows:
                    if not row or len(row) < 2:
                        continue
                    points, recognition = row[0], row[1]
                    if not recognition:
                        continue
                    text, confidence = str(recognition[0]).strip(), float(recognition[1])
                    if not text:
                        continue
                    coords = np.asarray(points, dtype=float)
                    left, top = int(coords[:, 0].min()), int(coords[:, 1].min())
                    right, bottom = int(coords[:, 0].max()), int(coords[:, 1].max())
                    words.append(WordBox(text, left, top, max(1, right-left), max(1, bottom-top), confidence * 100,
                                          block_num=len(words), par_num=0, line_num=len(words), word_num=0))
                    lines.append(text)
                    confidences.append(confidence * 100)
            # Paddle returns detected text lines; assign coherent reading
            # order to nearby boxes for downstream semantic field matching.
            # WordBox is frozen, so build updated copies instead of mutating.
            words.sort(key=lambda word: (word.center[1], word.left))
            line_id = 0
            previous_y = None
            relabeled: list[WordBox] = []
            for word in words:
                if previous_y is not None and abs(word.center[1] - previous_y) > max(12, word.height * 0.7):
                    line_id += 1
                relabeled.append(replace(word, block_num=0, par_num=0, line_num=line_id))
                previous_y = word.center[1]
            words = relabeled
            lines = [" ".join(w.text for w in sorted(words, key=lambda item: (item.line_num, item.left))
                                if w.line_num == idx) for idx in range(line_id + 1)]
            average = sum(confidences) / len(confidences) if confidences else 0.0
            status = "PARTIAL" if not words else ("LOW_CONFIDENCE" if average < settings.ocr_low_confidence_threshold else "COMPLETED")
            return OCRRawResult(self.name, self.version, "\n".join(lines), words, average, width, height,
                                (time.perf_counter()-start)*1000, status)
        except OCREngineUnavailableError:
            raise
        except Exception as exc:
            raise OCREngineUnavailableError(f"PaddleOCR inference failed: {exc}") from exc


_engine: PaddleOCREngine | None = None


def get_paddle_engine() -> PaddleOCREngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCREngine()
    return _engine
