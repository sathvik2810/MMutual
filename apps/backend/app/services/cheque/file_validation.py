"""File-level validation for uploaded cheques (docs/12_Cheque_Input_Module.md S5-S8).

Deliberately does not trust the filename extension alone (docs/12 S5:
"The system should not rely only on the filename extension; the actual
file type/content should also be validated") -- every accepted file is
checked against its magic-number signature and then actually decoded.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import PurePosixPath

import pymupdf
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.services.cheque.exceptions import (
    CorruptedFileError,
    EmptyUploadError,
    FileTooLargeError,
    InvalidPDFError,
    UnsupportedFileTypeError,
)

# Magic-number signatures for the three supported formats.
_JPEG_SIGNATURES = (b"\xff\xd8\xff",)
_PNG_SIGNATURES = (b"\x89PNG\r\n\x1a\n",)
_PDF_SIGNATURES = (b"%PDF-",)
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

_EXTENSION_TO_MEDIA_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    extension: str
    media_type: str
    file_size: int
    is_pdf: bool
    width: int | None  # None for PDFs (measured after rendering)
    height: int | None


def sanitize_filename(filename: str | None) -> str:
    """Strips any directory component so a crafted filename (e.g.
    "../../etc/passwd") cannot escape the upload storage directory
    (docs/36_Development_Guidelines.md S22: "Prevent path traversal")."""
    if not filename:
        return "upload"
    # PurePosixPath handles both "/" and left-over "\" fragments safely
    # since we only ever take .name (the final path component).
    name = PurePosixPath(filename.replace("\\", "/")).name
    return name or "upload"


def validate_extension(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_upload_extensions:
        raise UnsupportedFileTypeError()
    return ext


def validate_size(content: bytes) -> None:
    if not content:
        raise EmptyUploadError()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError()


def _matches_signature(content: bytes, signatures: tuple[bytes, ...]) -> bool:
    return any(content.startswith(sig) for sig in signatures)


def validate_content_matches_extension(content: bytes, extension: str) -> None:
    """Magic-number check -- catches a file renamed to a supported
    extension without actually being that format."""
    if extension in (".jpg", ".jpeg"):
        ok = _matches_signature(content, _JPEG_SIGNATURES)
    elif extension == ".png":
        ok = _matches_signature(content, _PNG_SIGNATURES)
    elif extension == ".pdf":
        ok = _matches_signature(content, _PDF_SIGNATURES)
    elif extension in (".tif", ".tiff"):
        ok = _matches_signature(content, _TIFF_SIGNATURES)
    else:  # pragma: no cover - unreachable, validate_extension runs first
        ok = False
    if not ok:
        raise UnsupportedFileTypeError()


def validate_image_integrity(content: bytes) -> tuple[int, int]:
    """Actually decodes the image (not just checks the header) so a
    truncated/corrupted file is caught here rather than later in the
    pipeline. Returns (width, height)."""
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CorruptedFileError() from exc

    if width < settings.min_hard_width_px or height < settings.min_hard_height_px:
        raise CorruptedFileError(
            "Image resolution is too low to process."
        )
    return width, height


def validate_pdf_integrity(content: bytes) -> int:
    """Opens the PDF and confirms it has at least one page. Per
    docs/12 S8, an MVP PDF should contain one cheque per request; if a
    multi-page PDF is uploaded, only the first page is used and this is
    reported back to the caller (not treated as an error)."""
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception as exc:  # pymupdf raises its own exception types
        raise InvalidPDFError() from exc

    if page_count < 1:
        raise InvalidPDFError()
    return page_count


def validate_upload(filename: str | None, content: bytes) -> ValidatedUpload:
    """Runs the complete file-validation pipeline described in
    docs/12_Cheque_Input_Module.md S5-S8, in order: extension -> size ->
    content-matches-extension -> integrity."""
    safe_filename = sanitize_filename(filename)
    extension = validate_extension(safe_filename)
    validate_size(content)
    validate_content_matches_extension(content, extension)

    is_pdf = extension == ".pdf"
    width = height = None
    if is_pdf:
        validate_pdf_integrity(content)
    else:
        width, height = validate_image_integrity(content)

    return ValidatedUpload(
        safe_filename=safe_filename,
        extension=extension,
        media_type=_EXTENSION_TO_MEDIA_TYPE[extension],
        file_size=len(content),
        is_pdf=is_pdf,
        width=width,
        height=height,
    )
