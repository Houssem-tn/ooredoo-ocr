import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = Path("temp_uploads")


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def validate_image_file(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
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

    print(f"[UPLOAD] filename={file.filename!r}  temp_path={file_path}")

    # file.read() consumes the stream; calling it again would return b"".
    # We read exactly once here and check immediately.
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with open(file_path, "wb") as f:
        f.write(content)

    written = file_path.stat().st_size
    print(f"[UPLOAD] written={written} bytes  exists={file_path.exists()}")

    if written == 0:
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
        pass