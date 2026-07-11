from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.ocr_engine import ocr_engine
from app.parser import detect_document_type, extract_structured_data
from app.schemas import BatchExtractResponse
from app.utils import delete_file_safely, save_temp_file


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


def extract(image_path: str) -> dict:
    ocr_result = ocr_engine.extract_best(image_path)

    best = ocr_result["best"]

    raw_text = best["raw_text"]
    ocr_items = best["items"]

    # extract_best pre-parses each pass (and, for router screens, merges the
    # EN and AR+EN passes per field) — re-parsing raw_text here would lose
    # the merge.
    extracted_data = best.get("structured")
    if extracted_data is None:
        extracted_data = extract_structured_data(
            raw_text,
            ocr_items=ocr_items,
            include_confidence=True,
        )

    return {
        "raw_ocr_text": raw_text,
        "extracted_data": extracted_data,
        "ocr_pass": best["variant"],
        "ocr_confidence": best["avg_confidence"],
        "ocr_passes": ocr_result["passes"],
    }


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

        result = extract(str(temp_path))
        raw_text = result["raw_ocr_text"]

        print(f"[BATCH] SUCCESS file={filename!r} confidence={result['ocr_confidence']}")

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
        print(f"[BATCH] FAILED file={filename!r}: {e.detail}")
        return _failed_result(filename, str(e.detail))

    except Exception as e:
        print(f"[BATCH] FAILED file={filename!r}: {e}")
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
    print(f"[BATCH] processing {total} image(s) case_id={case_id!r}")

    results: List[Dict[str, Any]] = []
    for index, file in enumerate(files, start=1):
        print(f"[BATCH] ({index}/{total}) file={file.filename!r}")
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