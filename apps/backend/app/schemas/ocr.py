from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OCRStartResponse(BaseModel):
    """docs/26_API_Specification.md S11 (POST /cheques/{id}/ocr)."""

    cheque_id: str
    status: str
    ocr_confidence: float
    engine: str


class ExtractedFieldResponse(BaseModel):
    """docs/15_Cheque_Data_Extraction.md S30 field-level metadata.

    The API surface is intentionally limited to these five documented
    keys; per-field reliability is derived internally from ``confidence``
    and is not part of the serialized contract.
    """

    value: Any
    raw_value: str | None
    confidence: float | None
    source: str | None
    validation_status: str | None


class OCRResultResponse(BaseModel):
    """docs/26_API_Specification.md S12 (GET /cheques/{id}/ocr), extended
    with the field-level raw/normalized/confidence metadata required by
    docs/15_Cheque_Data_Extraction.md."""

    cheque_id: str
    engine: str
    engine_version: str
    ocr_status: str
    confidence_score: float
    raw_text: str
    ocr_processing_time_ms: float
    extraction_status: str
    extraction_processing_time_ms: float
    missing_fields: list[str]
    ambiguous_fields: list[str]
    template: str
    signature_region_detected: bool
    signature_region_bbox: dict[str, int] | None
    extracted_data: dict[str, ExtractedFieldResponse]
    words: list[dict[str, Any]] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    extraction_warnings: list[str] = Field(default_factory=list)
    validation_results: dict[str, Any] = Field(default_factory=dict)
    status: str  # "SUCCESS" | "FAILED" (docs/26 S12 example)
    error_message: str | None = None
