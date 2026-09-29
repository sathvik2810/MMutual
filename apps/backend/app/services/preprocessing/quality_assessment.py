"""Image quality assessment (docs/13_Image_Preprocessing.md S16-S17).

Produces a structured, explainable result -- every factor is reported
individually with its measured value and pass/fail, not collapsed into an
opaque single score, per the project's explainability principle
(docs/36_Development_Guidelines.md S2).

All thresholds come from app.core.config (configurable, not hard-coded)
and are explicitly documented there as prototype defaults to be calibrated
experimentally -- matching docs/13's repeated caveat that "the exact
thresholds should be established experimentally using the project's
sample cheque dataset."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings


@dataclass
class QualityFactor:
    name: str
    value: float
    status: str  # "PASS" | "WARNING"
    message: str


@dataclass
class QualityResult:
    quality_status: str  # "ACCEPTABLE" | "POOR" | "UNSUITABLE"
    factors: list[QualityFactor] = field(default_factory=list)
    skew_angle_degrees: float = 0.0

    def as_dict(self) -> dict:
        return {
            "quality_status": self.quality_status,
            "skew_angle_degrees": round(self.skew_angle_degrees, 2),
            "factors": [
                {
                    "name": f.name,
                    "value": round(f.value, 2),
                    "status": f.status,
                    "message": f.message,
                }
                for f in self.factors
            ],
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def measure_blur(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness measure (docs/13 S17)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def measure_contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def measure_noise(gray: np.ndarray) -> float:
    """Simple noise proxy: mean absolute difference between the image and
    a median-blurred version of itself. Not a rigorous noise estimator,
    but sufficient to flag obviously noisy images for this prototype."""
    denoised = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, denoised)
    return float(diff.mean())


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimates the dominant skew angle in degrees using the minimum-area
    bounding rectangle of the thresholded foreground (docs/13 S12)."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 10:
        return 0.0
    (_, _), (rect_width, rect_height), raw_angle = cv2.minAreaRect(coords)
    # OpenCV versions differ in how minAreaRect represents a rotated box:
    # some return [-90, 0), others may return [0, 90). Convert the angle to
    # the long edge's orientation, then fold equivalent rectangle angles
    # into the small signed range around horizontal.
    angle = float(raw_angle)
    if rect_width < rect_height:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0
    elif angle > 45.0:
        angle -= 90.0
    return float(angle)


def assess_image_quality(image: np.ndarray) -> QualityResult:
    gray = _to_gray(image)
    height, width = gray.shape[:2]

    factors: list[QualityFactor] = []

    # Resolution
    if width < settings.min_soft_width_px or height < settings.min_soft_height_px:
        factors.append(QualityFactor(
            "resolution", float(min(width, height)), "WARNING",
            f"Resolution {width}x{height} is below the recommended minimum "
            f"({settings.min_soft_width_px}x{settings.min_soft_height_px}).",
        ))
    else:
        factors.append(QualityFactor("resolution", float(min(width, height)), "PASS",
                                      f"Resolution {width}x{height} is acceptable."))

    # Blur
    blur_score = measure_blur(gray)
    if blur_score < settings.blur_variance_threshold:
        factors.append(QualityFactor("blur", blur_score, "WARNING",
                                      "Image appears blurred (low sharpness measure)."))
    else:
        factors.append(QualityFactor("blur", blur_score, "PASS", "Image sharpness is acceptable."))

    # Brightness
    brightness = measure_brightness(gray)
    if brightness < settings.brightness_dark_threshold:
        factors.append(QualityFactor("brightness", brightness, "WARNING", "Image is too dark."))
    elif brightness > settings.brightness_bright_threshold:
        factors.append(QualityFactor("brightness", brightness, "WARNING", "Image is too bright/washed out."))
    else:
        factors.append(QualityFactor("brightness", brightness, "PASS", "Brightness is acceptable."))

    # Contrast
    contrast = measure_contrast(gray)
    if contrast < settings.contrast_enhancement_threshold:
        factors.append(QualityFactor("contrast", contrast, "WARNING", "Image has low contrast."))
    else:
        factors.append(QualityFactor("contrast", contrast, "PASS", "Contrast is acceptable."))

    # Skew
    skew_angle = estimate_skew_angle(gray)
    if abs(skew_angle) > settings.skew_correction_threshold_degrees:
        factors.append(QualityFactor("skew", skew_angle, "WARNING",
                                      f"Image is skewed by approximately {skew_angle:.1f} degrees."))
    else:
        factors.append(QualityFactor("skew", skew_angle, "PASS", "Image alignment is acceptable."))

    # Noise
    noise = measure_noise(gray)
    if noise > settings.noise_threshold:
        factors.append(QualityFactor("noise", noise, "WARNING", "Image contains significant noise."))
    else:
        factors.append(QualityFactor("noise", noise, "PASS", "Noise level is acceptable."))

    warning_count = sum(1 for f in factors if f.status == "WARNING")
    if width < settings.min_hard_width_px or height < settings.min_hard_height_px:
        quality_status = "UNSUITABLE"
    elif warning_count >= 3:
        quality_status = "UNSUITABLE"
    elif warning_count >= 1:
        quality_status = "POOR"
    else:
        quality_status = "ACCEPTABLE"

    return QualityResult(quality_status=quality_status, factors=factors, skew_angle_degrees=skew_angle)
