"""Cheque upload, detail, OCR, and validation endpoints.

Implements exactly the endpoints these milestones need from the canonical
contract (docs/26_API_Specification.md SS8-9, S11-S14): upload, get
details, run OCR, get OCR result, run validation, get validation result.
No fraud/decision endpoints are added here -- those belong to later
milestones.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.authorization import Permission, require_permission
from app.schemas.anomaly import AnomalyAnalysisResponse
from app.schemas.audit import AuditHistoryResponse
from app.schemas.cheque import ChequeDetailResponse, ChequeListResponse, ChequeSummaryResponse, ChequeUploadResponse
from app.schemas.decision import DecisionResponse
from app.schemas.fraud import FraudAnalysisResponse
from app.schemas.ocr import OCRResultResponse, OCRStartResponse
from app.schemas.risk import RiskScoreResponse
from app.schemas.signature import SignatureAnalysisResponse
from app.schemas.validation import ValidationResultResponse
from app.services.anomaly import anomaly_service
from app.services.anomaly.exceptions import ChequeNotExtractedForAnomalyError
from app.services.cheque import input_service, storage
from app.services.cheque.exceptions import ChequeInputError
from app.services.decision import decision_service
from app.services.decision.exceptions import RiskAssessmentNotAvailableError
from app.services.fraud import fraud_service
from app.services.fraud.exceptions import ChequeNotValidatedError
from app.services.ocr import pipeline as ocr_pipeline
from app.services.ocr.exceptions import ChequeNotPreprocessedError
from app.services.audit import audit_service
from app.services.risk import risk_service
from app.services.risk.exceptions import FraudAnalysisNotAvailableError
from app.services.signature import signature_service
from app.services.signature.exceptions import ChequeNotExtractedForSignatureError
from app.services.validation import validation_service
from app.services.validation.exceptions import ChequeNotExtractedError

router = APIRouter(prefix="/cheques", tags=["cheques"])


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """docs/26_API_Specification.md S36 error envelope."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": f"REQ-{uuid.uuid4().hex[:12]}"}},
    )


_ERROR_STATUS_CODES = {
    "EMPTY_UPLOAD": 400,
    "INVALID_FILE_TYPE": 415,
    "FILE_TOO_LARGE": 413,
    "CORRUPTED_FILE": 400,
    "INVALID_PDF": 400,
    "IMAGE_UNREADABLE": 400,
}


@router.post("/upload", status_code=201, response_model=ChequeUploadResponse)
async def upload_cheque(file: UploadFile = File(...)):
    content = await file.read()

    try:
        record = input_service.handle_upload(file.filename, content)
    except ChequeInputError as exc:
        status_code = _ERROR_STATUS_CODES.get(exc.code, 400)
        return JSONResponse(
            status_code=status_code,
            content={"success": False, "message": exc.message, "status": "REJECTED"},
        )

    return ChequeUploadResponse(
        success=True,
        cheque_id=record["cheque_id"],
        status="UPLOADED",
        message="Cheque uploaded successfully.",
    )


def _extracted_summary_fields(record: dict) -> dict:
    """Pulls the docs/26 S9 example's flat summary fields out of the
    extraction module's field-level structure (each field is
    {value, raw_value, confidence, ...}), without fabricating anything
    extraction never populated."""
    fields = (record.get("extraction") or {}).get("fields") or {}

    def _value(name: str):
        entry = fields.get(name)
        return entry.get("value") if isinstance(entry, dict) else None

    return {
        "cheque_number": _value("cheque_number"),
        "account_number": _value("account_number"),
        "routing_transit_number": _value("routing_transit_number"),
        "payee_name": _value("payee_name"),
        "amount": _value("amount"),
        "cheque_date": _value("date"),
    }


@router.get("", response_model=ChequeListResponse)
async def list_cheques(
    page: int = 1, limit: int = 20, status: str | None = None, risk_level: str | None = None,
    _role=Depends(require_permission(Permission.CHEQUE_VIEW)),
):
    """docs/26_API_Specification.md S10 (List Cheques) -- documented but
    not built by any earlier milestone; added here per this milestone's
    explicit gap-fill authorization (see the Milestone 9 report)."""
    records = input_service.list_cheque_records()
    if status:
        records = [r for r in records if r.get("processing_status") == status]
    if risk_level:
        records = [r for r in records if ((r.get("risk_assessment") or {}).get("risk_level")) == risk_level]

    records.sort(key=lambda r: r.get("upload_timestamp", ""), reverse=True)
    total = len(records)
    start = max(0, (page - 1) * limit)
    page_records = records[start : start + limit]

    summaries = []
    for record in page_records:
        decision = record.get("decision") or {}
        extracted = _extracted_summary_fields(record)
        summaries.append(ChequeSummaryResponse(
            cheque_id=record["cheque_id"],
            amount=extracted["amount"],
            risk_level=(record.get("risk_assessment") or {}).get("risk_level"),
            status=record.get("processing_status", "UNKNOWN"),
            payee_name=extracted["payee_name"],
            cheque_date=extracted["cheque_date"],
            upload_timestamp=record.get("upload_timestamp", ""),
            decision=decision.get("decision"),
        ))

    return ChequeListResponse(page=page, limit=limit, total=total, cheques=summaries)


@router.get("/{cheque_id}", response_model=ChequeDetailResponse)
async def get_cheque(cheque_id: str):
    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    return ChequeDetailResponse(**record, **_extracted_summary_fields(record))


@router.get("/{cheque_id}/image/{variant}")
async def get_cheque_image(cheque_id: str, variant: str):
    """Serves the original or processed cheque image (docs/13 S23 storage
    layout). Not part of docs/26's canonical endpoint table -- added here
    because Milestone 9 explicitly requires showing the original/processed
    image, and no earlier milestone exposed one over the API (images only
    ever lived on local disk). See the Milestone 9 report.

    Deliberately NOT gated behind require_permission: this project's only
    auth mechanism is the X-User-Role/X-User-Id dev-mode header pair
    (app/core/authorization.py), which an HTML <img> tag cannot attach --
    gating this endpoint would make it unusable from the browser without
    a real bearer-token/signed-URL scheme, which is out of scope for this
    MVP (see docs/29's own JWT-based target design). This mirrors the
    same unauthenticated state every other cheque-processing endpoint in
    this file is already in."""
    if variant not in ("original", "processed"):
        return _error_response(404, "NOT_FOUND", "Unknown image variant.")

    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")

    if variant == "processed":
        path = storage.processed_file_path(cheque_id)
    else:
        extension = Path(record["file_name"]).suffix.lower() or ".png"
        path = storage.original_file_path(cheque_id, extension)

    if not path.exists():
        return _error_response(404, "IMAGE_NOT_AVAILABLE", f"The {variant} image is not available for this cheque.")

    return FileResponse(path)


def _build_ocr_result_response(cheque_id: str, ocr: dict, extraction: dict) -> OCRResultResponse:
    outer_status = "FAILED" if ocr["ocr_status"] == "FAILED" else "SUCCESS"
    return OCRResultResponse(
        cheque_id=cheque_id,
        engine=ocr["engine_name"],
        engine_version=ocr["engine_version"],
        ocr_status=ocr["ocr_status"],
        confidence_score=ocr["average_confidence"],
        raw_text=ocr["raw_text"],
        ocr_processing_time_ms=ocr["processing_time_ms"],
        extraction_status=extraction["extraction_status"],
        extraction_processing_time_ms=extraction["processing_time_ms"],
        missing_fields=extraction["missing_fields"],
        ambiguous_fields=extraction["ambiguous_fields"],
        template=extraction["template"],
        signature_region_detected=extraction["signature_region_detected"],
        signature_region_bbox=extraction["signature_region_bbox"],
        extracted_data=extraction["fields"],
        words=ocr.get("words", []),
        image_width=ocr.get("image_width", 0),
        image_height=ocr.get("image_height", 0),
        extraction_warnings=extraction.get("warnings", []),
        validation_results=extraction.get("validation_results", {}),
        status=outer_status,
        error_message=ocr.get("error_message"),
    )


@router.post("/{cheque_id}/ocr", response_model=OCRStartResponse)
async def run_ocr(cheque_id: str):
    try:
        result = ocr_pipeline.run_ocr_and_extraction(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except ChequeNotPreprocessedError as exc:
        return _error_response(422, "CHEQUE_NOT_PREPROCESSED", str(exc))

    return OCRStartResponse(
        cheque_id=cheque_id,
        status=result["ocr"]["ocr_status"],
        ocr_confidence=result["ocr"]["average_confidence"],
        engine=result["ocr"]["engine_name"],
    )


@router.get("/{cheque_id}/ocr", response_model=OCRResultResponse)
async def get_ocr_result(cheque_id: str):
    result = ocr_pipeline.get_ocr_and_extraction(cheque_id)
    if result is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    if result["ocr"] is None:
        return _error_response(404, "OCR_NOT_RUN", "OCR has not been run for this cheque yet.")

    return _build_ocr_result_response(cheque_id, result["ocr"], result["extraction"])


@router.post("/{cheque_id}/validate", response_model=ValidationResultResponse)
async def run_validation(cheque_id: str):
    try:
        summary = validation_service.validate_cheque(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except ChequeNotExtractedError as exc:
        return _error_response(422, "CHEQUE_NOT_EXTRACTED", str(exc))

    return ValidationResultResponse(**summary.as_dict())


@router.get("/{cheque_id}/validation", response_model=ValidationResultResponse)
async def get_validation_result(cheque_id: str):
    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")

    validation = validation_service.get_validation_result(cheque_id)
    if validation is None:
        return _error_response(404, "VALIDATION_NOT_RUN", "Validation has not been run for this cheque yet.")

    return ValidationResultResponse(**validation)


def _build_fraud_analysis_response(payload: dict) -> FraudAnalysisResponse:
    tampering_detected = bool(payload.get("image_analysis", {}).get("image_tampering_score") or 0) and any(
        ind["type"] == "IMAGE_TAMPERING" for ind in payload.get("indicators", [])
    )
    return FraudAnalysisResponse(tampering_detected=tampering_detected, **payload)


@router.post("/{cheque_id}/fraud-analysis", response_model=FraudAnalysisResponse)
async def run_fraud_analysis(cheque_id: str):
    try:
        result = fraud_service.analyze_fraud(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except ChequeNotValidatedError as exc:
        return _error_response(422, "CHEQUE_NOT_VALIDATED", str(exc))

    payload = fraud_service.get_fraud_result(cheque_id)
    return _build_fraud_analysis_response(payload)


@router.get("/{cheque_id}/fraud-analysis", response_model=FraudAnalysisResponse)
async def get_fraud_analysis(cheque_id: str):
    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")

    payload = fraud_service.get_fraud_result(cheque_id)
    if payload is None:
        return _error_response(404, "FRAUD_ANALYSIS_NOT_RUN", "Fraud analysis has not been run for this cheque yet.")

    return _build_fraud_analysis_response(payload)


@router.post("/{cheque_id}/signature-analysis", response_model=SignatureAnalysisResponse)
async def run_signature_analysis(cheque_id: str):
    try:
        result = signature_service.analyze_signature(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except ChequeNotExtractedForSignatureError as exc:
        return _error_response(422, "CHEQUE_NOT_EXTRACTED", str(exc))

    return SignatureAnalysisResponse(**result.as_dict())


@router.post("/{cheque_id}/anomaly-analysis", response_model=AnomalyAnalysisResponse)
async def run_anomaly_analysis(cheque_id: str):
    try:
        result = anomaly_service.analyze_anomaly(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except ChequeNotExtractedForAnomalyError as exc:
        return _error_response(422, "CHEQUE_NOT_EXTRACTED", str(exc))

    return AnomalyAnalysisResponse(**result.as_dict())


@router.post("/{cheque_id}/risk-score", response_model=RiskScoreResponse)
async def run_risk_score(cheque_id: str):
    try:
        result = risk_service.calculate_risk(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except FraudAnalysisNotAvailableError as exc:
        return _error_response(422, "FRAUD_ANALYSIS_NOT_RUN", str(exc))

    return RiskScoreResponse(**result.as_dict())


@router.get("/{cheque_id}/audit", response_model=AuditHistoryResponse)
async def get_cheque_audit_history(cheque_id: str):
    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")

    events = audit_service.get_history(cheque_id)
    return AuditHistoryResponse(cheque_id=cheque_id, events=events)


@router.post("/{cheque_id}/decision", response_model=DecisionResponse)
async def run_decision(cheque_id: str):
    try:
        result = decision_service.make_decision(cheque_id)
    except KeyError:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")
    except RiskAssessmentNotAvailableError as exc:
        return _error_response(422, "RISK_ASSESSMENT_NOT_RUN", str(exc))

    return DecisionResponse(**result.as_dict())


@router.get("/{cheque_id}/decision", response_model=DecisionResponse)
async def get_decision(cheque_id: str):
    record = input_service.get_cheque_record(cheque_id)
    if record is None:
        return _error_response(404, "CHEQUE_NOT_FOUND", "The requested cheque does not exist.")

    payload = decision_service.get_decision(cheque_id)
    if payload is None:
        return _error_response(404, "DECISION_NOT_RUN", "A decision has not been made for this cheque yet.")

    return DecisionResponse(**payload)
