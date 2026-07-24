import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging_config import configure_logging, logger

# Must run before importing app.ocr_engine: that import instantiates
# EasyOCREngine() at module load time (see ocr_engine.py's bottom line),
# which logs as it loads each model — without this, those log calls would
# hit the default, unconfigured root logger (no handler, INFO dropped)
# instead of the real JSON/plain handler set up below.
configure_logging()

from app.metrics import (  # noqa: E402
    ocr_active_requests,
    ocr_request_duration_seconds,
    ocr_requests_total,
)
from app.ocr_engine import ocr_engine  # noqa: E402
from app.parser import detect_document_type, extract_structured_data  # noqa: E402
from app.request_context import get_request_id, new_request_id, set_request_id  # noqa: E402
from app.schemas import BatchExtractResponse  # noqa: E402
from app.utils import delete_file_safely, save_temp_file  # noqa: E402

app = FastAPI(
    title="EasyOCR Telecom Extraction API",
    version="2.0.0",
    description="OCR extraction API for telecom router signal screens and speedtest screenshots.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Accepts the caller's X-Request-ID (the backend's OcrClientService
    forwards the one it received from nginx/the client — see
    OBSERVABILITY_BACKUP.md "Correlation IDs") or mints one, makes it
    available to app.logging_config's log filter for the duration of the
    request, and echoes it back so the caller can confirm what OCR itself
    logged the request under.
    """

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        set_request_id(incoming or new_request_id())
        response = await call_next(request)
        response.headers["X-Request-ID"] = get_request_id()
        return response


app.add_middleware(RequestIdMiddleware)


@app.get("/metrics")
def metrics():
    # Internal-only — see app/metrics.py's own module docstring for why
    # this is safe to leave unauthenticated (never reachable outside the
    # cluster; the OCR Kubernetes Service is ClusterIP-only and the
    # Ingress never references it).
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def extract(image_path: str, endpoint: str = "extract") -> dict:
    """Shared OCR core for /extract, /extract/debug, and each file inside
    /extract/batch — metrics (request count, duration, active-in-flight)
    are recorded here once, rather than duplicated at each of those three
    call sites, since they all funnel through this one function.
    """
    ocr_active_requests.inc()
    started = time.monotonic()
    try:
        ocr_result = ocr_engine.extract_best(image_path)

        best = ocr_result["best"]

        raw_text = best["raw_text"]
        ocr_items = best["items"]

        # extract_best pre-parses each pass (and, for router screens, merges
        # the EN and AR+EN passes per field) — re-parsing raw_text here
        # would lose the merge.
        extracted_data = best.get("structured")
        if extracted_data is None:
            extracted_data = extract_structured_data(
                raw_text,
                ocr_items=ocr_items,
                include_confidence=True,
            )

        ocr_requests_total.labels(endpoint=endpoint, status="success").inc()
        return {
            "raw_ocr_text": raw_text,
            "extracted_data": extracted_data,
            "ocr_pass": best["variant"],
            "ocr_confidence": best["avg_confidence"],
            "ocr_passes": ocr_result["passes"],
        }
    except HTTPException:
        ocr_requests_total.labels(endpoint=endpoint, status="invalid_input").inc()
        raise
    except Exception:
        ocr_requests_total.labels(endpoint=endpoint, status="failure").inc()
        raise
    finally:
        ocr_request_duration_seconds.labels(endpoint=endpoint).observe(time.monotonic() - started)
        ocr_active_requests.dec()


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "EasyOCR Telecom Extraction API",
        "version": "2.0.0",
    }


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path = await save_temp_file(file)

        result = extract(str(temp_path), endpoint="extract")

        raw_text = result["raw_ocr_text"]
        data = result["extracted_data"]
        document_type = detect_document_type(raw_text)

        return {
            "success": True,
            "documentType": document_type,
            "data": data,
            "rawText": raw_text,
            "ocrPass": result["ocr_pass"],
            "ocrConfidence": result["ocr_confidence"],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {str(e)}",
        )

    finally:
        if temp_path is not None:
            delete_file_safely(temp_path)


def _failed_result(filename: str, error: str) -> Dict[str, Any]:
    return {
        "filename": filename,
        "status": "FAILED",
        "documentType": None,
        "data": None,
        "rawText": None,
        "ocrPass": None,
        "ocrConfidence": None,
        "error": error,
    }


async def process_single_attachment(file: UploadFile) -> Dict[str, Any]:
    """Run OCR on one uploaded attachment and return a per-image result.

    Reuses the same EasyOCR pipeline as the single-file endpoint. This never
    raises: validation/OCR failures are captured and reported as
    status=FAILED so a batch can continue with the remaining files.
    """
    filename = file.filename or "unknown"
    temp_path = None

    try:
        # save_temp_file validates the extension (JPG/JPEG/PNG) and rejects
        # empty uploads, raising HTTPException on bad input.
        temp_path = await save_temp_file(file)

        result = extract(str(temp_path), endpoint="extract_batch")
        raw_text = result["raw_ocr_text"]

        logger.info("batch item succeeded", extra={"filename": filename, "confidence": result["ocr_confidence"]})

        return {
            "filename": filename,
            "status": "SUCCESS",
            "documentType": detect_document_type(raw_text),
            "data": result["extracted_data"],
            "rawText": raw_text,
            "ocrPass": result["ocr_pass"],
            "ocrConfidence": result["ocr_confidence"],
            "error": None,
        }

    except HTTPException as e:
        logger.warning("batch item failed", extra={"filename": filename, "detail": str(e.detail)})
        return _failed_result(filename, str(e.detail))

    except Exception as e:
        logger.warning("batch item failed", extra={"filename": filename, "error": str(e)})
        return _failed_result(filename, str(e))

    finally:
        if temp_path is not None:
            delete_file_safely(temp_path)


async def process_multiple_images(
    files: List[UploadFile],
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Loop over uploaded attachments, OCR each one, keep results separate.

    EasyOCR handles one image at a time, so files are processed sequentially.
    A failure on one image is isolated to that image's result.
    """
    total = len(files)
    logger.info("batch processing started", extra={"total": total, "case_id": case_id})

    results: List[Dict[str, Any]] = []
    for index, file in enumerate(files, start=1):
        logger.info("batch item processing", extra={"index": index, "total": total, "filename": file.filename})
        results.append(await process_single_attachment(file))

    return results


@app.post("/extract/batch", response_model=BatchExtractResponse)
async def extract_batch_endpoint(
    files: List[UploadFile] = File(...),
    caseId: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = await process_multiple_images(files, caseId)
    succeeded = sum(1 for r in results if r["status"] == "SUCCESS")

    return {
        "success": True,
        "caseId": caseId,
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }


@app.post("/extract/debug")
async def extract_debug_endpoint(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path = await save_temp_file(file)

        result = extract(str(temp_path))

        raw_text = result["raw_ocr_text"]
        data = result["extracted_data"]
        document_type = detect_document_type(raw_text)

        return {
            "success": True,
            "documentType": document_type,
            "data": data,
            "rawText": raw_text,
            "ocrPass": result["ocr_pass"],
            "ocrConfidence": result["ocr_confidence"],
            "ocrPasses": result["ocr_passes"],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR debug extraction failed: {str(e)}",
        )

    finally:
        if temp_path is not None:
            delete_file_safely(temp_path)