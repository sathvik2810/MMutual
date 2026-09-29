"""Application configuration.

Loads non-secret defaults from the shared/environment YAML files under
``config/`` and layers environment-variable-based overrides (via
``pydantic-settings``) on top for anything sensitive (database URL, secret
key). Secrets must never be committed to the repository; they are supplied
through a local ``.env`` file (see ``.env.example``) or real environment
variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/backend/app/core/config.py -> repo root is four levels up
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(environment: str) -> dict[str, Any]:
    """Merge config/settings.yaml with the environment-specific override file."""
    base = _load_yaml(CONFIG_DIR / "settings.yaml")
    override = _load_yaml(CONFIG_DIR / f"{environment}.yaml")
    return _deep_merge(base, override)


class Settings(BaseSettings):
    """Environment-variable-backed settings. Never hard-code secrets here."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    app_name: str = "Mass Mutual Cheque Fraud Detection System"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/mass_mutual_db"
    secret_key: str = "changeme-local-development-only"

    # --- Milestone 8: Database connection pool (docs/24 S2, ADR-0003) ---
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: int = 30
    database_connect_timeout_seconds: int = 3
    # When no reachable PostgreSQL is configured, the application falls
    # back to the in-memory/CSV repositories already used by Milestones
    # 3-7 (see app/repositories/repository_factory.py) rather than
    # failing every request -- this is an explicit, documented fallback
    # for environments without a live PostgreSQL server, not a silent
    # substitute database engine (no SQLite is ever used).
    use_postgres_repositories: bool = True

    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    max_upload_size_mb: int = 10

    # --- Milestone 2: Cheque Input & Image Preprocessing --------------
    # These are intentionally configuration fields (not hard-coded inside
    # the processing logic), per docs/36_Development_Guidelines.md S9 and
    # docs/12_Cheque_Input_Module.md S6 -- all are explicitly documented as
    # "example prototype rules" to be calibrated later, not fixed banking
    # standards.
    allowed_upload_extensions: list[str] = [".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff"]

    # Below this size an image is treated as unreadable/invalid (hard
    # rejection). Below the "soft" size the image is accepted but flagged
    # as poor quality (docs/12 S7: "reject / request better image").
    min_hard_width_px: int = 80
    min_hard_height_px: int = 40
    min_soft_width_px: int = 500
    min_soft_height_px: int = 220

    # Laplacian-variance sharpness threshold; below this the image is
    # flagged as blurred (docs/13 S17). Calibratable per docs' own caveat.
    # Calibrated against the Milestone 1 synthetic cheque images (clean
    # samples measure ~2500); a threshold two orders of magnitude below
    # that comfortably separates genuinely blurred test images.
    blur_variance_threshold: float = 150.0

    # Mean pixel intensity (0-255) bounds for "too dark"/"too bright".
    # Calibrated against the Milestone 1 synthetic cheque images: their
    # light cream background measures ~mean 240-250, which is normal for
    # a document photo/scan and must not be flagged as "washed out".
    brightness_dark_threshold: float = 90.0
    brightness_bright_threshold: float = 251.0

    # Standard-deviation-of-intensity threshold below which contrast
    # enhancement (CLAHE) is applied. Calibrated so the Milestone 1
    # synthetic cheques (mostly white background, sparse text -> std
    # dev ~28-30) read as acceptable, while a genuinely low-contrast
    # (e.g. faded) image still triggers enhancement.
    contrast_enhancement_threshold: float = 22.0

    # Skew angle (degrees) beyond which deskew rotation is applied.
    skew_correction_threshold_degrees: float = 1.0

    # Simple noise proxy (mean abs diff vs. median-blurred version);
    # above this the image is flagged as noisy.
    noise_threshold: float = 6.0

    # Controlled upscaling target when an image is smaller than this on
    # its longer side but still above the hard-minimum resolution.
    target_working_long_side_px: int = 1600
    max_upscale_factor: float = 2.5

    # --- Milestone 3: OCR & Cheque Data Extraction ---------------------
    # PaddleOCR is the primary OCR adapter. Tesseract remains installed as
    # a native executable and is called through PyTesseract when PaddleOCR
    # fails or returns insufficient text. Its binary path is configurable.
    tesseract_cmd_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Below this average confidence (Tesseract's own 0-100 scale) the OCR
    # result is treated as LOW_CONFIDENCE and a retry with an alternate
    # image representation is attempted (docs/14 S16-S17). Explicitly a
    # calibratable prototype default, not a proven accuracy threshold.
    ocr_low_confidence_threshold: float = 55.0

    # Below this per-field confidence, an individual extracted field is
    # flagged low-confidence even if the overall OCR run succeeded.
    ocr_field_low_confidence_threshold: float = 40.0

    # Minimum fraction of non-background pixels in the signature region
    # for it to be reported as "signature detected" (docs/15 S21). This is
    # region *detection* only -- not comparison/verification.
    signature_region_ink_ratio_threshold: float = 0.01

    # --- Milestone 4: Validation Engine ---------------------------------
    # Business thresholds, per docs/16_Validation_Engine.md's repeated
    # instruction that these must be configuration, not hard-coded inside
    # validation functions (S16 date validity; S21 amount limits).
    mock_banking_data_dir: str = "data/mock_banking_data"

    # A cheque older than this many days (relative to the processing
    # date) is treated as stale/expired (docs/16 S16). Matches the
    # assumption documented in scripts/generate_synthetic_data.py so the
    # Milestone 1 STALE_CHEQUE category is evaluated consistently.
    cheque_validity_period_days: int = 180

    # Whether a cheque dated after the processing date should be allowed
    # to PASS date validation (docs/16 S15). Default false: future-dated
    # cheques fail DATE_WINDOW.
    allow_future_dated_cheques: bool = False

    # Maximum permitted cheque amount (docs/16 S21). Matches the
    # `amount_limit` used in the Milestone 1 mock cheque-issuance registry.
    max_permitted_cheque_amount: float = 500000.00

    # Per-check severities (docs/16 S27: "Severity values should be
    # configurable"). Defaults follow docs/16 S27's own example table
    # where it gives one explicitly (e.g. stopped cheque = CRITICAL,
    # payee mismatch = HIGH); checks the table doesn't cover are assigned
    # a documented, reasonable default severity (see Milestone 4 report).
    validation_severities: dict[str, str] = {
        "REQUIRED_FIELDS": "HIGH",
        "ACCOUNT_EXISTS": "HIGH",
        "ACCOUNT_STATUS": "HIGH",
        "CHEQUE_SERIES": "MEDIUM",
        "CHEQUE_STATUS": "HIGH",
        "CHEQUE_STATUS_STOPPED": "CRITICAL",
        "DATE_WINDOW": "MEDIUM",
        "PAYEE_MATCH": "HIGH",
        "ROUTING_TRANSIT": "MEDIUM",
        "AMOUNT": "HIGH",
        "AMOUNT_CONSISTENCY": "HIGH",
        "DUPLICATE_CHECK": "HIGH",
        "CROSS_FIELD": "MEDIUM",
    }

    # --- Milestone 5: Fraud Detection Engine ----------------------------
    # docs/17_Fraud_Detection.md S21 documents the fraud engine's own
    # internal risk composition (image/validation/duplicate/anomaly/
    # transaction risk) as distinct from the later, separate Risk Scoring
    # Module (docs/21, Milestone 6/7) that additionally folds in
    # signature and OCR-confidence risk -- these weights are the fraud
    # engine's own prototype configuration, not the final risk formula.
    fraud_score_weights: dict[str, float] = {
        "tampering": 30.0,
        "validation_signal": 25.0,
        "duplicate": 30.0,
        "pattern": 15.0,
    }

    # 0-100 fraud_risk_score bands (docs/17 S21, initial project
    # thresholds -- not proven banking thresholds).
    fraud_risk_bands: dict[str, list[int]] = {
        "LOW": [0, 29],
        "MEDIUM": [30, 59],
        "HIGH": [60, 79],
        "CRITICAL": [80, 100],
    }

    # RULE-005: number of HIGH+CRITICAL-severity fraud indicators that
    # triggers a mandatory manual-review escalation regardless of the
    # numeric score (docs/17 S19).
    fraud_multi_indicator_review_threshold: int = 2

    # Image tampering detector (docs/17 S7-S9): per-region statistics are
    # compared against the cheque's own background reference rather than
    # any external/trained baseline. Thresholds are explicitly prototype
    # values pending calibration against a larger dataset.
    tampering_regions: list[str] = ["amount", "payee_name", "date", "cheque_number"]
    tampering_noise_zscore_threshold: float = 2.5
    tampering_edge_zscore_threshold: float = 2.5
    tampering_score_review_threshold: float = 0.5

    # Multi-level duplicate detection (docs/19 S12-S16): prototype
    # similarity thresholds, explicitly calibratable.
    duplicate_perceptual_hash_size: int = 8
    duplicate_confirmed_similarity_threshold: float = 0.95
    duplicate_potential_similarity_threshold: float = 0.80

    # Pattern/behavioral indicators (docs/20 S9-S13): initial statistical
    # thresholds, not banking-industry norms.
    pattern_amount_zscore_threshold: float = 3.0
    pattern_min_history_for_amount_baseline: int = 3
    pattern_frequency_window_days: int = 2
    pattern_frequency_baseline_days: int = 30
    pattern_frequency_ratio_threshold: float = 3.0
    pattern_cheque_sequence_gap_threshold: int = 50

    fraud_engine_version: str = "fraud-engine-v1.0-rule-based"

    # Number of FAILed Milestone 4 validation checks that constitutes
    # "multiple validation failures" as its own fraud indicator (docs/17
    # S18/S1 indicator #12).
    validation_multiple_failures_threshold: int = 2

    # --- Milestone 6: Signature Analysis ---------------------------------
    # docs/18_Signature_Analysis.md S12 Method 2 (feature-based
    # comparison). Calibration note (Milestone 6 report): raw pixel/NCC
    # similarity between this project's procedurally-generated synthetic
    # signatures was measured to carry almost no discriminative signal
    # (genuine-vs-own-variation and genuine-vs-unrelated-forged scored in
    # the same noisy band), because each reference is an independently
    # random stroke pattern with no shared underlying "handwriting
    # identity". Aggregate structural features (ink density, stroke/
    # connected-component count, bounding-box extent) showed a real, if
    # weak, measured separation instead, so the comparator is
    # feature-based rather than pixel-similarity-based.
    # Measured against the real Milestone 1 signature test dataset: sharp
    # (genuine/partial/altered) samples measure blur-variance in the
    # thousands (~5,000-15,000); the deliberately blurred "low_quality"
    # sample measures ~22. 100.0 sits comfortably between the two.
    signature_quality_blur_variance_threshold: float = 100.0
    signature_quality_min_ink_pixel_count: int = 40
    signature_feature_weights: dict[str, float] = {
        "density": 1.0, "component_count": 1.0, "bbox_width": 1.0, "bbox_height": 1.0, "aspect_ratio": 1.0,
    }
    # Similarity = 1 / (1 + weighted_feature_distance / scale). Calibrated
    # against this project's own measured genuine/forged distributions
    # (Milestone 6 report), not the illustrative 0.85/0.70/0.50 example
    # values in docs/18 S14, which assume much stronger separation than
    # this synthetic dataset actually exhibits.
    signature_similarity_distance_scale: float = 20.0
    signature_risk_thresholds: dict[str, float] = {"LOW": 0.55, "MEDIUM": 0.45, "HIGH": 0.35}
    signature_engine_version: str = "signature-v1.0-feature-based"

    # --- Milestone 6: Anomaly Detection -----------------------------------
    # Reuses the same statistical thresholds as Milestone 5's
    # pattern-indicator detectors (pattern_amount_zscore_threshold etc.)
    # since both analyze the identical underlying statistics; this module
    # additionally owns the combined weighted anomaly_score (docs/20 S20).
    anomaly_weights: dict[str, float] = {
        "amount": 30.0, "frequency": 20.0, "payee": 20.0, "sequence": 10.0, "timing": 10.0, "transaction_pattern": 10.0,
    }
    # Amount Z-score severity bands. docs/20 S11's illustrative example
    # (|Z| 2-3 Moderate, 3-4 High, >4 Critical) was measured against this
    # project's own dataset (Milestone 6 report) and found far too
    # sensitive: transactions.csv's historical amounts are drawn from a
    # systematically narrower/lower range than actual cheque amounts
    # (transaction mean ~$10.7k vs cheque mean ~$18.6k), so the naive
    # bands flagged 63% of ALL cheques -- including most VALID ones --
    # as at least "Moderate". These recalibrated cutoffs were chosen
    # because they measured 0% false positives on the real VALID-category
    # sample while still separating genuinely extreme cases (e.g. the
    # MULTIPLE_ANOMALIES category, whose cheques are deliberately amount-
    # tampered, stayed correctly flagged at every tested cutoff).
    anomaly_amount_zscore_bands: dict[str, float] = {"MODERATE": 5.0, "HIGH": 8.0, "CRITICAL": 12.0}
    anomaly_risk_bands: dict[str, list[int]] = {
        "LOW": [0, 24], "MEDIUM": [25, 49], "HIGH": [50, 74], "CRITICAL": [75, 100],
    }
    anomaly_engine_version: str = "anomaly-v1.0-rule-based-statistical"

    # --- Milestone 6: Overall Risk Scoring ---------------------------------
    # docs/21_Risk_Scoring.md S7 weight table -- deliberately DISTINCT from
    # Milestone 5's own fraud_score_weights/fraud_risk_bands (docs/17 S21):
    # this is the separate, later-stage Risk Scoring Engine that combines
    # fraud, validation, signature, duplicate, anomaly, and OCR-confidence
    # signals, per docs/25's separate fraud_results/risk_assessments tables.
    risk_factor_weights: dict[str, float] = {
        "tampering": 20.0, "signature": 20.0, "duplicate": 20.0, "anomaly": 20.0,
        "validation": 10.0, "ocr": 5.0, "other": 5.0,
    }
    risk_bands: dict[str, list[int]] = {
        "LOW": [0, 24], "MEDIUM": [25, 49], "HIGH": [50, 74], "CRITICAL": [75, 100],
    }
    risk_tampering_contribution_bands: dict[str, float] = {"NONE": 0, "LOW": 5, "MODERATE": 10, "HIGH": 15, "STRONG": 20}
    # image_tampering_score (0.00-1.00) cut points below which the NONE/LOW/MODERATE/HIGH band applies (docs/21 S8 gives labels only, no numeric cutoffs).
    risk_tampering_score_cutoffs: dict[str, float] = {"NONE": 0.15, "LOW": 0.30, "MODERATE": 0.50, "HIGH": 0.75}
    risk_signature_contribution_bands: dict[str, float] = {"LOW": 0, "MEDIUM": 10, "HIGH": 15, "CRITICAL": 20, "UNAVAILABLE": 0}
    risk_duplicate_contribution_bands: dict[str, float] = {"NEW": 0, "POTENTIAL_DUPLICATE": 10, "CONFIRMED_DUPLICATE": 20}
    risk_validation_contribution_bands: dict[str, float] = {"PASS": 0, "WARNING": 3, "FAIL": 10}
    risk_ocr_confidence_contribution_bands: dict[str, float] = {"95": 0, "85": 1, "70": 3, "0": 5}
    risk_config_version: str = "risk-v1.0"

    # --- Milestone 7: Decision Engine -------------------------------------
    # docs/22_Decision_Engine.md S9 hard-rule/priority hierarchy. Priority
    # 3 (risk-score fallback) deliberately REUSES Milestone 6's own
    # risk_bands/risk_level (LOW/MEDIUM/HIGH/CRITICAL) rather than
    # re-deriving numeric cutoffs a second time -- risk_bands already
    # equals docs/22 S20's example thresholds (0-24/25-49/50-74/75-100).
    decision_min_ocr_confidence: float = 70.0  # below this: MANDATORY_REVIEW (docs/22 S9 Priority 2 "insufficient_OCR_confidence")
    decision_policy_version: str = "decision-policy-v1.0"
    decision_engine_version: str = "decision-v1.0"

    # --- Milestone 7: Manual Review Workflow ------------------------------
    review_priority_by_risk_level: dict[str, str] = {"CRITICAL": "CRITICAL", "HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
    review_workflow_version: str = "review-v1.0"

    def yaml_config(self) -> dict[str, Any]:
        return load_yaml_config(self.environment)

    @property
    def runtime_data_dir(self) -> Path:
        return REPO_ROOT / "data" / "runtime"

    @property
    def runtime_original_dir(self) -> Path:
        return self.runtime_data_dir / "original"

    @property
    def runtime_processed_dir(self) -> Path:
        return self.runtime_data_dir / "processed"

    @property
    def mock_banking_data_path(self) -> Path:
        return REPO_ROOT / self.mock_banking_data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
