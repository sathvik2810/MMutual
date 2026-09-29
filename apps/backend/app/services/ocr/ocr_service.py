"""OCR orchestration (docs/14_OCR_Engine.md S6, S16-S17).

Runs the configured OCREngine against Milestone 2's processed image. If
the result comes back LOW_CONFIDENCE, retries once against the original
(unprocessed) image as an alternate representation -- a concrete, simple
version of the documented "Attempt 1 standard preprocessing -> OCR ->
low confidence -> Attempt 2 enhanced preprocessing -> OCR -> compare
results" strategy (docs/14 S17), bounded to a single retry so it cannot
threaten the <30s/cheque budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.services.ocr.engine import OCREngine, OCRRawResult
from app.services.ocr.tesseract_engine import get_ocr_engine, get_tesseract_engine
from app.services.preprocessing.preprocessing_service import load_image_bgr


@dataclass
class OCRRunOutcome:
    result: OCRRawResult
    attempts: int
    total_processing_time_ms: float


def run_ocr_for_cheque(
    processed_image_path: str,
    original_image_path: str,
    *,
    engine: OCREngine | None = None,
) -> OCRRunOutcome:
    engine = engine or get_ocr_engine()
    start = time.perf_counter()
    attempts = 0

    try:
        processed_image = load_image_bgr(processed_image_path)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        failed = OCRRawResult(
            engine_name=engine.name, engine_version=engine.version, raw_text="",
            status="FAILED", error_message=f"Unable to load processed image: {exc}",
            processing_time_ms=elapsed_ms,
        )
        return OCRRunOutcome(result=failed, attempts=0, total_processing_time_ms=elapsed_ms)

    errors: list[str] = []
    result = None
    attempts += 1
    try:
        result = engine.run(processed_image)
        if result.status in ("FAILED", "PARTIAL", "LOW_CONFIDENCE") or not result.raw_text.strip() or len(result.words) < 2:
            errors.append(result.error_message or "Primary OCR output was empty or insufficient.")
            result = None
    except Exception as exc:  # Paddle can fail during import, model loading, or inference.
        errors.append(str(exc))

    if result is None and engine.name != "Tesseract":
        fallback = get_tesseract_engine()
        try:
            attempts += 1
            result = fallback.run(processed_image)
            if errors:
                result.error_message = "Primary OCR fallback: " + "; ".join(errors)
        except Exception as exc:
            errors.append(str(exc))
    elif result is None:
        # A caller may explicitly inject Tesseract for isolated adapter use.
        result = OCRRawResult(engine.name, engine.version, "", status="FAILED", error_message="; ".join(errors))

    if result is None:
        fallback = get_tesseract_engine()
        result = OCRRawResult(fallback.name, "unavailable", "", status="FAILED",
                              error_message="; ".join(errors) or "Both OCR engines failed.")

    if result.engine_name == "Tesseract" and result.status == "LOW_CONFIDENCE":
        try:
            original_image = load_image_bgr(original_image_path)
            retry_result = get_tesseract_engine().run(original_image)
            attempts += 1
            if retry_result.average_confidence > result.average_confidence:
                result = retry_result
        except Exception:  # noqa: BLE001 - retry is best-effort; keep the first result on any failure
            pass

    total_elapsed_ms = (time.perf_counter() - start) * 1000
    return OCRRunOutcome(result=result, attempts=attempts, total_processing_time_ms=total_elapsed_ms)
