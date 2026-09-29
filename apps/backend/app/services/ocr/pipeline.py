"""Ties the cheque repository, OCR engine, and extraction service
together for the /cheques/{id}/ocr endpoints. Kept separate from
ocr_service.py (which only knows about running the OCR engine itself) so
that engine invocation stays independently testable from repository/
persistence concerns.
"""

from __future__ import annotations

from pathlib import Path

from app.repositories.cheque_repository import get_cheque_repository
from app.services.audit import audit_service
from app.services.extraction.extraction_service import extract_cheque_data
from app.services.ocr.exceptions import ChequeNotPreprocessedError
from app.services.ocr.ocr_service import run_ocr_for_cheque
from app.services.preprocessing.preprocessing_service import load_image_bgr


def run_ocr_and_extraction(cheque_id: str) -> dict:
    repo = get_cheque_repository()
    record = repo.get(cheque_id)
    if record is None:
        raise KeyError(cheque_id)

    preprocessing = record.get("preprocessing") or {}
    if preprocessing.get("preprocessing_status") != "COMPLETED" or not preprocessing.get("processed_image_path"):
        raise ChequeNotPreprocessedError(
            "Cheque has not completed preprocessing yet; OCR cannot run."
        )

    processed_image_path = preprocessing["processed_image_path"]
    original_extension = Path(record["file_name"]).suffix.lower() or ".png"
    from app.services.cheque.storage import original_file_path
    original_path = original_file_path(cheque_id, original_extension)

    outcome = run_ocr_for_cheque(processed_image_path, str(original_path))
    ocr_result = outcome.result

    processed_image = load_image_bgr(processed_image_path)
    extraction_result = extract_cheque_data(cheque_id, ocr_result, processed_image)

    ocr_dict = {
        "engine_used": ocr_result.engine_name,
        "engine_name": ocr_result.engine_name,
        "engine_version": ocr_result.engine_version,
        "raw_text": ocr_result.raw_text,
        "average_confidence": round(ocr_result.average_confidence, 2),
        "ocr_status": ocr_result.status,
        "attempts": outcome.attempts,
        "processing_time_ms": round(outcome.total_processing_time_ms, 2),
        "error_message": ocr_result.error_message,
        "words": [
            {"text": w.text, "bbox": {"x": w.left, "y": w.top, "width": w.width, "height": w.height},
             "confidence": w.confidence}
            for w in ocr_result.words
        ],
        "image_width": ocr_result.image_width,
        "image_height": ocr_result.image_height,
    }
    extraction_dict = extraction_result.as_dict()

    new_status = "OCR_COMPLETED" if ocr_result.status != "FAILED" else "FAILED"
    repo.update(cheque_id, {
        "ocr": ocr_dict,
        "extraction": extraction_dict,
        "processing_status": new_status,
    })
    audit_service.record(
        event_type="OCR_COMPLETED", cheque_id=cheque_id, source="SYSTEM",
        new_status=new_status, action="RUN_OCR", result=ocr_result.status,
        metadata={"confidence_score": ocr_dict["average_confidence"]},
    )

    return {"ocr": ocr_dict, "extraction": extraction_dict}


def get_ocr_and_extraction(cheque_id: str) -> dict | None:
    record = get_cheque_repository().get(cheque_id)
    if record is None:
        return None
    return {"ocr": record.get("ocr"), "extraction": record.get("extraction")}
