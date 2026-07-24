import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np
from easyocr.utils import reformat_input
from fastapi import HTTPException

from app.logging_config import logger
from app.metrics import ocr_invalid_input_total, ocr_model_loaded
from app.parser import (
    extract_structured_data,
    merge_router_results,
    router_critical_fields_valid,
)

# Optimal width for EasyOCR on label/screenshot images.
# Phone photos are 3000–4000px wide — running OCR on the full resolution
# multiplies inference time by 4–9x with no accuracy gain.
# 1600px gives EasyOCR enough resolution to read small label text reliably.
_OCR_TARGET_WIDTH = 1600


class EasyOCREngine:
    def __init__(self):
        logger.info("loading EasyOCR models")
        self.reader_en = easyocr.Reader(["en"], gpu=False)
        # Recognition-only: detection boxes always come from reader_en
        # (detection is language-agnostic), so this reader never needs
        # its own detection model.
        self.reader_ar_en = easyocr.Reader(["ar", "en"], gpu=False, detector=False)
        ocr_model_loaded.set(1)
        logger.info("EasyOCR models loaded")

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def extract_best(self, image_path: str) -> Dict[str, Any]:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_size = path.stat().st_size
        logger.info("processing image", extra={"size_bytes": file_size})

        # cv2.imread returns None on Windows for paths with non-ASCII chars
        # or truncated files — fall back to imdecode-from-bytes.
        image = cv2.imread(str(path))
        if image is None:
            logger.info("cv2.imread returned None, trying imdecode fallback")
            raw = path.read_bytes()
            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None:
            ocr_invalid_input_total.labels(reason="unreadable_image").inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable image file",
            )

        if image.size == 0 or len(image.shape) < 2:
            ocr_invalid_input_total.labels(reason="empty_or_corrupted_image").inc()
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty or corrupted",
            )

        # Normalise to target width BEFORE OCR.
        # This is the primary speed fix: a 4000px phone photo becomes 1600px,
        # reducing pixel count by ~6x and OCR time from 2 min to ~15 s on CPU.
        image = self._normalize_size(image)
        logger.debug("normalized image", extra={"shape": str(image.shape)})

        # Detect text boxes once — detection (CRAFT) is language-agnostic.
        # Only recognition differs per language, so both passes share boxes.
        img_grey, horizontal_list, free_list = self._detect_boxes(image)

        # EN-only recognition first. The AR+EN recognition model is worse on
        # Latin text (misreads "dB" as "d8"/"d3"), so the EN result is
        # authoritative for every field it can fill.
        items, raw_text, avg_confidence = self._recognize(
            self.reader_en, img_grey, horizontal_list, free_list
        )

        if not self._looks_like_router_signal(raw_text) and self._looks_like_equipment_label(raw_text):
            # Barcode decoding is fast (milliseconds) and improves MAC/IMEI accuracy
            # for equipment labels. QR codes are never decoded (credential safety).
            for bc in self._decode_barcodes(image):
                items.append({
                    "text": bc["value"],
                    "confidence": 1.0,
                    "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "source": "barcode",
                    "barcode_type": bc["type"],
                })

        structured = extract_structured_data(
            raw_text, ocr_items=items, include_confidence=True
        )

        en_pass = {
            "variant": "en_only",
            "raw_text": raw_text,
            "items": items,
            "avg_confidence": round(avg_confidence, 4),
            "score": round(avg_confidence * 10, 4),
            "structured": structured,
        }

        if structured.get("type") != "router_signal_screen" or router_critical_fields_valid(structured):
            # Equipment labels and speedtests never use the Arabic reader, and
            # a router screen the EN pass read completely doesn't need it either.
            logger.debug("en pass sufficient, skipping ar+en pass")
            return {"best": en_pass, "passes": [en_pass]}

        # AR+EN fallback: critical fields are missing — recognize the SAME
        # boxes with the Arabic+English model and fill only the gaps.
        logger.debug("critical fields missing in en pass, running ar+en fallback")
        ar_items, ar_text, ar_confidence = self._recognize(
            self.reader_ar_en, img_grey, horizontal_list, free_list
        )
        ar_structured = extract_structured_data(
            ar_text, ocr_items=ar_items, include_confidence=True
        )
        if ar_structured.get("type") != "router_signal_screen":
            ar_structured = None

        ar_pass = {
            "variant": "ar_en_fallback",
            "raw_text": ar_text,
            "items": ar_items,
            "avg_confidence": round(ar_confidence, 4),
            "score": round(ar_confidence * 10, 4),
            "structured": ar_structured,
        }

        # Per-field merge: every valid EN field is kept, the AR+EN pass only
        # fills the missing/invalid ones. EN never gets overwritten.
        merged = merge_router_results(structured, ar_structured)

        best = {
            "variant": "en_primary_ar_en_fallback",
            "raw_text": raw_text,
            "items": items,
            "avg_confidence": round(avg_confidence, 4),
            "score": round(avg_confidence * 10, 4),
            "structured": merged,
        }

        return {
            "best": best,
            "passes": [en_pass, ar_pass],
        }

    # ---------------------------------------------------------------
    # Image size normalisation
    # ---------------------------------------------------------------

    @staticmethod
    def _normalize_size(image: np.ndarray) -> np.ndarray:
        """Resize to _OCR_TARGET_WIDTH using INTER_AREA (downscale) or INTER_CUBIC (upscale)."""
        h, w = image.shape[:2]
        if abs(w - _OCR_TARGET_WIDTH) < 100:
            return image  # already close — skip the warp
        scale = _OCR_TARGET_WIDTH / w
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (_OCR_TARGET_WIDTH, new_h), interpolation=interp)

    # ---------------------------------------------------------------
    # OCR runner
    # ---------------------------------------------------------------

    def _detect_boxes(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[Any], List[Any]]:
        """Run text detection once; return the grey image and text boxes.

        Detection (CRAFT) is language-agnostic — reader_en's detector serves
        every recognition pass, so the image is never detected twice.
        """
        img, img_cv_grey = reformat_input(image)
        horizontal_list, free_list = self.reader_en.detect(img, reformat=False)
        # detect() returns one list per image in the batch — we pass exactly one
        return img_cv_grey, horizontal_list[0], free_list[0]

    def _recognize(
        self,
        reader,
        img_grey: np.ndarray,
        horizontal_list: List[Any],
        free_list: List[Any],
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        results = reader.recognize(
            img_grey,
            horizontal_list,
            free_list,
            detail=1,
            paragraph=False,
            reformat=False,
        )
        items: List[Dict[str, Any]] = []
        for bbox, text, confidence in results:
            cleaned = text.strip()
            if cleaned:
                items.append(
                    {"text": cleaned, "confidence": float(confidence), "bbox": bbox}
                )
        raw_text = self.clean_text("\n".join(item["text"] for item in items))
        avg_confidence = (
            sum(item["confidence"] for item in items) / len(items)
            if items
            else 0.0
        )
        return items, raw_text, avg_confidence

    # ---------------------------------------------------------------
    # Equipment label detection
    # ---------------------------------------------------------------

    @staticmethod
    def _looks_like_router_signal(text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in ("imsi", "rsrp", "rsrq", "earfcn", "sinr"))

    @staticmethod
    def _looks_like_equipment_label(text: str) -> bool:
        low = text.lower()
        has_mac = bool(re.search(r"\bmac\b", low))
        has_sn = bool(re.search(r"\bs/n\b|\bserial\b", low))
        has_brand = "tozed" in low or "zlt" in low or "wireless router" in low
        return (has_mac and has_sn) or (has_brand and (has_mac or has_sn))

    # ---------------------------------------------------------------
    # Barcode decoding (1D only — QR explicitly blocked)
    # ---------------------------------------------------------------

    def _decode_barcodes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Decode 1D barcodes. QR codes are never decoded (credential safety)."""
        results: List[Dict[str, Any]] = []

        try:
            detector = cv2.barcode.BarcodeDetector()
            ok, decoded_info, decoded_type, _ = detector.detectAndDecodeWithType(image)
            if ok and decoded_info:
                for info, btype in zip(decoded_info, decoded_type):
                    if not info or "QR" in str(btype).upper():
                        continue
                    self._classify_barcode_value(info.strip(), results)
        except (AttributeError, cv2.error, Exception):
            pass

        try:
            from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore
            for sym in pyzbar_decode(image):
                if sym.type.upper() == "QRCODE":
                    continue
                data = sym.data.decode("utf-8", errors="ignore").strip()
                self._classify_barcode_value(data, results)
        except ImportError:
            pass

        return results

    @staticmethod
    def _classify_barcode_value(
        value: str, results: List[Dict[str, Any]]
    ) -> None:
        """Append to results only if value matches MAC, IMEI, or SN pattern."""
        norm = re.sub(r"[:\-\s]", "", value).upper()
        if re.fullmatch(r"[0-9A-F]{12}", norm):
            results.append({"value": norm, "type": "mac"})
            return
        if re.fullmatch(r"\d{14,17}", value.strip()):
            results.append({"value": value.strip(), "type": "imei"})
            return
        if re.fullmatch(r"[A-Z][0-9]{2}[A-Z0-9]{5,16}", value.upper()):
            results.append({"value": value.upper(), "type": "sn"})

    # ---------------------------------------------------------------
    # Text cleaning
    # ---------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace(",", ".")
        text = text.replace("：", ":")
        text = text.replace("−", "-")
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()


ocr_engine = EasyOCREngine()
