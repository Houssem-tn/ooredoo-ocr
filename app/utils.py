import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.logging_config import logger
from app.metrics import ocr_invalid_input_total, ocr_temp_file_cleanup_failures_total

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = Path("temp_uploads")


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def validate_image_file(file: UploadFile) -> str:
    if not file.filename:
        ocr_invalid_input_total.labels(reason="missing_filename").inc()
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        ocr_invalid_input_total.labels(reason="unsupported_extension").inc()
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG files are allowed",
        )

    return ext


def generate_unique_filename(extension: str) -> str:
    return f"{uuid.uuid4().hex}{extension}"


async def save_temp_file(file: UploadFile) -> Path:
    ensure_upload_dir()

    extension = validate_image_file(file)
    filename = generate_unique_filename(extension)
    file_path = UPLOAD_DIR / filename

    # Generated (uuid4) temp filename only — never the caller's original
    # filename, which is somewhat identifying/attacker-controlled input.
    logger.info("saving upload", extra={"temp_path": str(file_path)})

    # file.read() consumes the stream; calling it again would return b"".
    # We read exactly once here and check immediately.
    content = await file.read()

    if not content:
        ocr_invalid_input_total.labels(reason="empty_file").inc()
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with open(file_path, "wb") as f:
        f.write(content)

    written = file_path.stat().st_size
    logger.debug("upload written", extra={"written_bytes": written})

    if written == 0:
        ocr_invalid_input_total.labels(reason="write_failed").inc()
        raise HTTPException(
            status_code=400,
            detail="File was not written to disk correctly",
        )

    return file_path


def delete_file_safely(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except Exception:
        ocr_temp_file_cleanup_failures_total.inc()
        logger.warning("failed to delete temp file", extra={"temp_path": str(path)})