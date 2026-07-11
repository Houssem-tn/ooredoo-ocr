"""
Tests for the EN-first / AR-as-fallback OCR strategy.

Covers:
  - normalize_db_noise unit-token repair (d8/d3/dO -> dB, digit+0B -> dB)
  - extract_metric_pair on AR+EN-corrupted strings
  - per-metric range validation + needsReview flagging
  - router_critical_fields_valid gate
  - merge_router_results per-field merge (EN never overwritten)
  - extract_best flow: EN pass returned immediately when clean, AR+EN pass
    only runs as fallback, equipment label / speedtest never use AR

Engine-level tests stub the EasyOCR readers — no models are loaded.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.parser import (
    extract_metric_pair,
    merge_router_results,
    metric_pair_in_range,
    normalize_db_noise,
    parse_router_screen,
    router_critical_fields_valid,
)

# easyocr.Reader is patched during import so the module-level singleton
# doesn't load CRAFT/recognition models.
with patch("easyocr.Reader"):
    from app.ocr_engine import EasyOCREngine

from tests.test_router_ocr import ARABIC_OCR, EQUIPMENT_LABEL_OCR, FRENCH_OCR


# What the EN-only reader produces for an Arabic router screen: Latin labels
# read cleanly, the Arabic SINR/RSRP labels and values come out as junk.
EN_OCR_OF_ARABIC_SCREEN = """IMSI
605030964873447
IMEI
861321070112543
PCI
147+147+147/364
puda Jl
Juw> Jl>
RSSI
-69dBm/-100dBm
EARFCN
1800+6400+101/628704
RSRQ
-16.0dB/-11dB
Cell ID
083B80B/-"""

FRENCH_OCR_BAD_RSRQ = FRENCH_OCR.replace("-14.0dB/-11dB", "8dB/-11dB")

SPEEDTEST_OCR = """SPEEDTEST
Descendant
Ascendant
93.46
14.97
Mbps"""


# ---------------------------------------------------------------------------
# normalize_db_noise
# ---------------------------------------------------------------------------

class TestNormalizeDbNoise:
    def test_corrupted_sinr(self):
        assert normalize_db_noise("3. 5d3/1 50B") == "3.5dB/15dB"

    def test_corrupted_rsrq(self):
        assert normalize_db_noise("-14.0d8/-11d8") == "-14.0dB/-11dB"

    def test_corrupted_dbm(self):
        assert normalize_db_noise("-75d8m/-73d3m") == "-75dBm/-73dBm"

    def test_d_misread_as_capital_o(self):
        assert normalize_db_noise("3.5dO/15dO") == "3.5dB/15dB"

    def test_digit_ob_token(self):
        assert normalize_db_noise("-11OB") == "-11dB"

    def test_clean_db_unchanged(self):
        assert normalize_db_noise("3.5dB/15dB") == "3.5dB/15dB"

    def test_clean_dbm_unchanged(self):
        assert normalize_db_noise("-75dBm/-73dBm") == "-75dBm/-73dBm"

    def test_arabic_units_unchanged(self):
        line = "-0.5 ديسيبل /11 ديسيبل"
        assert normalize_db_noise(line) == line


# ---------------------------------------------------------------------------
# extract_metric_pair on corrupted strings (acceptance test 2)
# ---------------------------------------------------------------------------

class TestExtractMetricPairCorrupted:
    def test_sinr_corrupted(self):
        assert extract_metric_pair("3. 5d3/1 50B", "dB") == "3.5dB/15dB"

    def test_rsrq_corrupted(self):
        assert extract_metric_pair("-14.0d8/-11d8", "dB") == "-14.0dB/-11dB"

    def test_rsrp_corrupted(self):
        assert extract_metric_pair("-75d8m/-73d8m", "dBm") == "-75dBm/-73dBm"

    def test_clean_values_still_verbatim(self):
        assert extract_metric_pair("3.5dB/15dB", "dB") == "3.5dB/15dB"
        assert extract_metric_pair("-14.0dB/-11dB", "dB") == "-14.0dB/-11dB"


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------

class TestMetricPairInRange:
    def test_valid_pairs(self):
        assert metric_pair_in_range("sinr", "3.5dB/15dB") is True
        assert metric_pair_in_range("rsrp", "-75dBm/-73dBm") is True
        assert metric_pair_in_range("rssi", "-60dBm/-62dBm") is True
        assert metric_pair_in_range("rsrq", "-14.0dB/-11dB") is True

    def test_positive_rsrq_rejected(self):
        assert metric_pair_in_range("rsrq", "8dB/-11dB") is False

    def test_implausible_rsrp_rejected(self):
        assert metric_pair_in_range("rsrp", "-30dBm/-73dBm") is False

    def test_no_range_defined(self):
        assert metric_pair_in_range("pci", "487+487+487/876") is None

    def test_none_pair(self):
        assert metric_pair_in_range("sinr", None) is None


class TestRangeValidationInParse:
    """Acceptance test 4: 8dB/-11dB RSRQ must be rejected and flagged."""

    def setup_method(self):
        self.result = parse_router_screen(FRENCH_OCR_BAD_RSRQ)

    def test_range_valid_flag(self):
        assert self.result["rsrq"]["rangeValid"] is False

    def test_confidence_capped(self):
        assert self.result["rsrq"]["confidence"] <= 0.4

    def test_needs_review(self):
        assert self.result["needsReview"] is True

    def test_other_fields_untouched(self):
        assert self.result["sinr"]["value"] == "3.5dB/15dB"
        assert self.result["sinr"]["rangeValid"] is True
        assert self.result["sinr"]["confidence"] > 0.4

    def test_valid_screen_has_range_flags(self):
        result = parse_router_screen(FRENCH_OCR)
        for metric in ("sinr", "rsrp", "rssi", "rsrq"):
            assert result[metric]["rangeValid"] is True
        assert result["needsReview"] is False


class TestOcrItemConfidenceCap:
    """Field confidence is capped at the OCR confidence of the source item."""

    @staticmethod
    def _items(text, confidence):
        return [
            {"text": line, "confidence": confidence,
             "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]]}
            for line in text.splitlines() if line.strip()
        ]

    def test_low_item_confidence_caps_field(self):
        result = parse_router_screen(
            FRENCH_OCR, ocr_items=self._items(FRENCH_OCR, 0.42)
        )
        assert result["sinr"]["confidence"] == 0.42
        assert result["sinr"]["confidenceLevel"] == "low"

    def test_high_item_confidence_keeps_rule_score(self):
        result = parse_router_screen(
            FRENCH_OCR, ocr_items=self._items(FRENCH_OCR, 0.97)
        )
        assert result["sinr"]["confidence"] == 0.92

    def test_no_items_keeps_rule_score(self):
        result = parse_router_screen(FRENCH_OCR)
        assert result["sinr"]["confidence"] == 0.92


# ---------------------------------------------------------------------------
# router_critical_fields_valid
# ---------------------------------------------------------------------------

class TestRouterCriticalFieldsValid:
    def test_complete_french_screen(self):
        assert router_critical_fields_valid(parse_router_screen(FRENCH_OCR)) is True

    def test_complete_arabic_screen(self):
        assert router_critical_fields_valid(parse_router_screen(ARABIC_OCR)) is True

    def test_missing_fields(self):
        partial = parse_router_screen("IMSI\n605030968128782")
        assert router_critical_fields_valid(partial) is False

    def test_out_of_range_field(self):
        bad = parse_router_screen(FRENCH_OCR_BAD_RSRQ)
        assert router_critical_fields_valid(bad) is False

    def test_non_router_type(self):
        assert router_critical_fields_valid({"type": "EQUIPMENT_LABEL"}) is False


# ---------------------------------------------------------------------------
# merge_router_results
# ---------------------------------------------------------------------------

class TestMergeRouterResults:
    def test_valid_en_never_overwritten(self):
        en = parse_router_screen(FRENCH_OCR)
        ar = parse_router_screen(ARABIC_OCR)
        merged = merge_router_results(en, ar)
        for field in ("imsi", "imei", "pci", "sinr", "rsrp", "rssi", "earfcn", "rsrq"):
            assert merged[field]["value"] == en[field]["value"], field
            assert merged[field]["source"].endswith("@en_pass"), field
        assert merged["needsReview"] is False

    def test_missing_en_fields_filled_from_ar(self):
        en = parse_router_screen(EN_OCR_OF_ARABIC_SCREEN)
        ar = parse_router_screen(ARABIC_OCR)
        merged = merge_router_results(en, ar)
        # SINR/RSRP were unreadable for the EN pass — filled from AR+EN
        assert merged["sinr"]["value"] == "-0.5dB/11dB"
        assert merged["sinr"]["source"] == "scoped_fallback@ar_en_pass"
        assert merged["rsrp"]["value"] == "-87dBm/-112dBm"
        assert merged["rsrp"]["source"] == "scoped_fallback@ar_en_pass"
        # Everything the EN pass read stays EN
        assert merged["imsi"]["value"] == "605030964873447"
        assert merged["imsi"]["source"].endswith("@en_pass")
        assert merged["rssi"]["value"] == "-69dBm/-100dBm"
        assert merged["rssi"]["source"].endswith("@en_pass")
        assert merged["needsReview"] is False

    def test_out_of_range_en_replaced_by_valid_ar(self):
        en = parse_router_screen(FRENCH_OCR_BAD_RSRQ)
        ar = parse_router_screen(ARABIC_OCR)
        merged = merge_router_results(en, ar)
        assert merged["rsrq"]["value"] == "-16.0dB/-11dB"
        assert merged["rsrq"]["source"].endswith("@ar_en_pass")

    def test_out_of_range_in_both_passes_flags_review(self):
        en = parse_router_screen(FRENCH_OCR_BAD_RSRQ)
        ar = parse_router_screen(
            ARABIC_OCR.replace("-16.0 ديسيبل /-11 ديسيبل", "8 ديسيبل /-11 ديسيبل")
        )
        merged = merge_router_results(en, ar)
        assert merged["rsrq"]["value"] == "8dB/-11dB"
        assert merged["rsrq"]["confidence"] <= 0.4
        assert merged["needsReview"] is True

    def test_merge_without_ar_result(self):
        en = parse_router_screen(EN_OCR_OF_ARABIC_SCREEN)
        merged = merge_router_results(en, None)
        assert merged["sinr"]["value"] is None
        assert merged["needsReview"] is True


# ---------------------------------------------------------------------------
# extract_best flow (stubbed readers)
# ---------------------------------------------------------------------------

def _items_from_text(text, confidence):
    return [
        {"text": line, "confidence": confidence,
         "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        for line in text.splitlines() if line.strip()
    ]


def _make_engine(en_payload, ar_payload, calls):
    """EasyOCREngine with stubbed detection/recognition — no models needed."""
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.reader_en = "EN_READER"
    engine.reader_ar_en = "AR_READER"
    engine._detect_boxes = lambda image: (None, [], [])
    engine._decode_barcodes = lambda image: []

    def fake_recognize(reader, img_grey, horizontal_list, free_list):
        calls.append(reader)
        return en_payload if reader == "EN_READER" else ar_payload

    engine._recognize = fake_recognize
    return engine


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "screen.png"
    cv2.imwrite(str(path), np.zeros((40, 1600, 3), dtype=np.uint8))
    return str(path)


class TestExtractBestEnFirst:
    """Acceptance test 1: clean French screen never triggers the Arabic pass."""

    def _run(self, image_path):
        calls = []
        engine = _make_engine(
            en_payload=(_items_from_text(FRENCH_OCR, 0.95), FRENCH_OCR, 0.95),
            ar_payload=([], "", 0.0),
            calls=calls,
        )
        return engine.extract_best(image_path), calls

    def test_arabic_pass_not_run(self, image_path):
        _, calls = self._run(image_path)
        assert calls == ["EN_READER"]

    def test_single_en_pass_returned(self, image_path):
        result, _ = self._run(image_path)
        assert result["best"]["variant"] == "en_only"
        assert len(result["passes"]) == 1

    def test_en_values_verbatim(self, image_path):
        result, _ = self._run(image_path)
        structured = result["best"]["structured"]
        assert structured["sinr"]["value"] == "3.5dB/15dB"
        assert structured["rsrq"]["value"] == "-14.0dB/-11dB"
        assert structured["rsrp"]["value"] == "-75dBm/-73dBm"
        assert structured["imsi"]["value"] == "605030968128782"
        assert structured["needsReview"] is False

    def test_en_sources_unmangled(self, image_path):
        result, _ = self._run(image_path)
        assert result["best"]["structured"]["sinr"]["source"] == "label_rule"


class TestExtractBestArabicFallback:
    """Acceptance test 3: Arabic screen fills missing fields via AR+EN pass."""

    def _run(self, image_path):
        calls = []
        engine = _make_engine(
            en_payload=(
                _items_from_text(EN_OCR_OF_ARABIC_SCREEN, 0.9),
                EN_OCR_OF_ARABIC_SCREEN,
                0.9,
            ),
            ar_payload=(_items_from_text(ARABIC_OCR, 0.62), ARABIC_OCR, 0.62),
            calls=calls,
        )
        return engine.extract_best(image_path), calls

    def test_both_passes_run_en_first(self, image_path):
        _, calls = self._run(image_path)
        assert calls == ["EN_READER", "AR_READER"]

    def test_merged_variant_and_passes(self, image_path):
        result, _ = self._run(image_path)
        assert result["best"]["variant"] == "en_primary_ar_en_fallback"
        assert len(result["passes"]) == 2

    def test_missing_fields_filled_from_ar(self, image_path):
        result, _ = self._run(image_path)
        structured = result["best"]["structured"]
        assert structured["sinr"]["value"] == "-0.5dB/11dB"
        assert structured["sinr"]["source"].endswith("@ar_en_pass")
        assert structured["rsrp"]["value"] == "-87dBm/-112dBm"

    def test_en_fields_kept(self, image_path):
        result, _ = self._run(image_path)
        structured = result["best"]["structured"]
        assert structured["rssi"]["value"] == "-69dBm/-100dBm"
        assert structured["rssi"]["source"].endswith("@en_pass")
        assert structured["imsi"]["value"] == "605030964873447"

    def test_no_review_needed_after_merge(self, image_path):
        result, _ = self._run(image_path)
        assert result["best"]["structured"]["needsReview"] is False


class TestExtractBestNonRouterFlows:
    """Acceptance test 5: equipment label and speedtest never use the AR reader."""

    def test_equipment_label_stays_en_only(self, image_path):
        calls = []
        engine = _make_engine(
            en_payload=(
                _items_from_text(EQUIPMENT_LABEL_OCR, 0.9),
                EQUIPMENT_LABEL_OCR,
                0.9,
            ),
            ar_payload=([], "", 0.0),
            calls=calls,
        )
        result = engine.extract_best(image_path)
        assert calls == ["EN_READER"]
        assert result["best"]["structured"]["type"] == "EQUIPMENT_LABEL"
        assert result["best"]["structured"]["macAddress"]["value"] == "98A942FF5EE9"

    def test_speedtest_stays_en_only(self, image_path):
        calls = []
        engine = _make_engine(
            en_payload=(_items_from_text(SPEEDTEST_OCR, 0.9), SPEEDTEST_OCR, 0.9),
            ar_payload=([], "", 0.0),
            calls=calls,
        )
        result = engine.extract_best(image_path)
        assert calls == ["EN_READER"]
        assert result["best"]["structured"]["type"] == "speedtest_screen"
