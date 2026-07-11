from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExtractResponse(BaseModel):
    success: bool
    documentType: str
    data: Dict[str, Any]
    rawText: str


class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None


class ImageOCRResult(BaseModel):
    """OCR outcome for a single uploaded attachment in a batch.

    ``status`` is ``SUCCESS`` or ``FAILED``. On failure only ``filename``,
    ``status`` and ``error`` are populated; the OCR fields stay ``None``.
    """

    filename: str
    status: str
    documentType: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    rawText: Optional[str] = None
    ocrPass: Optional[str] = None
    ocrConfidence: Optional[float] = None
    error: Optional[str] = None


class BatchExtractResponse(BaseModel):
    success: bool
    caseId: Optional[str] = None
    total: int
    succeeded: int
    failed: int
    results: List[ImageOCRResult]