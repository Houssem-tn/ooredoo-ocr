"""
Unit tests for router_signal_screen OCR extraction.

Tests run against raw OCR text strings — no image or EasyOCR needed.
Both French/English and Arabic/mixed label variants are covered.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.parser import (
    detect_document_type,
    normalize_arabic,
    normalize_arabic_units,
    normalize_signal_metric,
    extract_metric_pair,
    extract_imsi_imei,
    extract_pci,
    extract_earfcn,
    extract_cell_id,
    parse_router_screen,
    parse_equipment_label,
    field_result,
    confidence_level,
)


# ---------------------------------------------------------------------------
# Raw OCR text fixtures
# ---------------------------------------------------------------------------

FRENCH_OCR = """IMSI
605030968128782
IMEI
861321072113879
PCI
487+487+487/876
SINR
3.5dB/15dB
RSRP
-75dBm/-73dBm
RSSI
-60dBm/-62dBm
EARFCN
101+1800+6400/628704
RSRQ
-14.0dB/-11dB
Cell ID
075C415/-"""

# Real Arabic OCR output as produced by EasyOCR ["ar", "en"].
# Arabic labels for SINR and RSRP are deliberately unrecognized (corrupted) to
# exercise the scoped-block fallback (Pass 3a).
ARABIC_OCR = """IMSI
605030964873447
IMEI رقم
861321070112543
PCI
147+147+147/364
مؤشر الإشارة
-0.5 ديسيبل /11 ديسيبل
مستوى الاستقبال
-87 ديسيبل واط / -112 ديسيبل واط
RSSI
-69 ديسيبل /-100 ديسيبل
EARFCN
1800+6400+101/628704
RSRQ
-16.0 ديسيبل /-11 ديسيبل
خلايا
083B80B/-"""

# Arabic OCR where the RSRP value is split across multiple lines.
# "سعر التجزئة" is a corrupted/different RSRP label (not in aliases).
# The value -87 and -112 each live on separate lines alongside Arabic unit words.
ARABIC_OCR_MULTILINE_RSRP = """IMSI
605030964873447
IMEI رقم
861321070112543
PCI
147+147+147/364
مؤشر الإشارة
-0.5 ديسيبل /11 ديسيبل
سعر التجزئة
واط
واط / -112 ديسيبل
-87 ديسيبل
RSSI
-69 ديسيبل /-100 ديسيبل
EARFCN
1800+6400+101/628704
RSRQ
-16.0 ديسيبل /-11 ديسيبل
خلايا
083B80B/-"""

# Same layout, but IMSI label comes through as "IMS|" (OCR pipe/slash confusion).
ARABIC_OCR_IMSI_NOISE = """IMS|
605030964873447
IMEI رقم
861321070112543
PCI
147+147+147/364
مؤشر الإشارة
-0.5 ديسيبل /11 ديسيبل
مستوى الاستقبال
-87 ديسيبل واط / -112 ديسيبل واط
RSSI
-69 ديسيبل /-100 ديسيبل
EARFCN
1800+6400+101/628704
RSRQ
-16.0 ديسيبل /-11 ديسيبل
خلايا
083B80B/-"""


# ---------------------------------------------------------------------------
# normalize_arabic
# ---------------------------------------------------------------------------

class TestNormalizeArabic:
    def test_alef_variants(self):
        assert normalize_arabic("أإآا") == "اااا"

    def test_ya_variant(self):
        assert normalize_arabic("ى") == "ي"

    def test_ta_marbuta(self):
        assert normalize_arabic("ة") == "ه"

    def test_arabic_indic_digits(self):
        assert normalize_arabic("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    def test_eastern_arabic_digits(self):
        assert normalize_arabic("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    def test_sinr_label_normalization(self):
        raw = "نسبة الإشارة إلى الضوضاء والتداخل"
        norm = normalize_arabic(raw)
        assert "نسبه" in norm
        assert "الاشاره" in norm
        assert "الي" in norm


# ---------------------------------------------------------------------------
# normalize_arabic_units
# ---------------------------------------------------------------------------

class TestNormalizeArabicUnits:
    def test_plain_db(self):
        assert normalize_arabic_units("ديسيبل") == "dB"

    def test_db_with_number(self):
        result = normalize_arabic_units("-0.5 ديسيبل")
        assert "dB" in result
        assert "-0.5" in result

    def test_dbm_watt(self):
        result = normalize_arabic_units("ديسيبل واط")
        assert result == "dBm"

    def test_dbm_milli_watt(self):
        result = normalize_arabic_units("ديسيبل ملي واط")
        assert result == "dBm"

    def test_dbm_meeli_watt(self):
        result = normalize_arabic_units("ديسيبل ميلي واط")
        assert result == "dBm"

    def test_standalone_milli_watt(self):
        result = normalize_arabic_units("ملي واط")
        assert result == "dBm"

    def test_sinr_full_line(self):
        line = "-0.5 ديسيبل /11 ديسيبل"
        result = normalize_arabic_units(line)
        assert "dB" in result
        assert "-0.5" in result
        assert "11" in result
        assert "ديسيبل" not in result

    def test_rsrp_full_line(self):
        line = "-87 ديسيبل واط / -112 ديسيبل واط"
        result = normalize_arabic_units(line)
        assert "dBm" in result
        assert "-87" in result
        assert "-112" in result
        assert "ديسيبل" not in result


# ---------------------------------------------------------------------------
# normalize_signal_metric
# ---------------------------------------------------------------------------

class TestNormalizeSignalMetric:
    def test_sinr_arabic(self):
        result = normalize_signal_metric("-0.5 ديسيبل /11 ديسيبل", "sinr")
        assert result == "-0.5dB/11dB"

    def test_rsrp_arabic(self):
        result = normalize_signal_metric("-87 ديسيبل واط / -112 ديسيبل واط", "rsrp")
        assert result == "-87dBm/-112dBm"

    def test_rssi_arabic(self):
        result = normalize_signal_metric("-69 ديسيبل /-100 ديسيبل", "rssi")
        assert result == "-69dBm/-100dBm"

    def test_rsrq_arabic(self):
        result = normalize_signal_metric("-16.0 ديسيبل /-11 ديسيبل", "rsrq")
        assert result == "-16.0dB/-11dB"

    def test_sinr_french(self):
        assert normalize_signal_metric("3.5dB/15dB", "sinr") == "3.5dB/15dB"

    def test_rsrp_french(self):
        assert normalize_signal_metric("-75dBm/-73dBm", "rsrp") == "-75dBm/-112dBm".replace("-112", "-73")

    def test_none_input(self):
        assert normalize_signal_metric(None, "sinr") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_metric_pair
# ---------------------------------------------------------------------------

class TestExtractMetricPair:
    def test_french_sinr(self):
        assert extract_metric_pair("3.5dB/15dB", "dB") == "3.5dB/15dB"

    def test_french_rsrp(self):
        assert extract_metric_pair("-75dBm/-73dBm", "dBm") == "-75dBm/-73dBm"

    def test_french_rsrq(self):
        assert extract_metric_pair("-14.0dB/-11dB", "dB") == "-14.0dB/-11dB"

    def test_arabic_db_format(self):
        result = extract_metric_pair("-0.5 ديسيبل /11 ديسيبل", "dB")
        assert result == "-0.5dB/11dB"

    def test_arabic_dbm_format(self):
        result = extract_metric_pair("-87 ديسيبل واط / -112 ديسيبل واط", "dBm")
        assert result == "-87dBm/-112dBm"

    def test_arabic_rssi(self):
        result = extract_metric_pair("-69 ديسيبل /-100 ديسيبل", "dBm")
        assert result == "-69dBm/-100dBm"

    def test_none_input(self):
        assert extract_metric_pair(None, "dB") is None

    def test_empty_input(self):
        assert extract_metric_pair("", "dB") is None


# ---------------------------------------------------------------------------
# Individual field extractors
# ---------------------------------------------------------------------------

class TestExtractors:
    def test_imsi(self):
        assert extract_imsi_imei("605030968128782") == "605030968128782"

    def test_imei(self):
        assert extract_imsi_imei("861321072113879") == "861321072113879"

    def test_pci_compound(self):
        assert extract_pci("487+487+487/876") == "487+487+487/876"

    def test_earfcn(self):
        assert extract_earfcn("1800+6400+101/628704") == "1800+6400+101/628704"

    def test_cell_id_french(self):
        result = extract_cell_id("075C415/-")
        assert result is not None
        assert "075C415" in result.upper()

    def test_cell_id_arabic(self):
        result = extract_cell_id("083B80B/-")
        assert result is not None
        assert "083B80B" in result.upper()


# ---------------------------------------------------------------------------
# confidence_level thresholds
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    def test_high(self):
        assert confidence_level(0.92) == "high"
        assert confidence_level(0.85) == "high"

    def test_medium_at_0_6(self):
        assert confidence_level(0.60) == "medium"

    def test_medium_at_0_55(self):
        assert confidence_level(0.55) == "medium"

    def test_medium_at_0_70(self):
        assert confidence_level(0.70) == "medium"

    def test_low_below_threshold(self):
        assert confidence_level(0.50) == "low"
        assert confidence_level(0.0) == "low"


# ---------------------------------------------------------------------------
# detect_document_type
# ---------------------------------------------------------------------------

class TestDetectDocumentType:
    def test_french_router(self):
        assert detect_document_type(FRENCH_OCR) == "router_signal_screen"

    def test_arabic_router(self):
        assert detect_document_type(ARABIC_OCR) == "router_signal_screen"

    def test_arabic_router_by_arabic_dB_word(self):
        text = "IMSI\n605030964873447\n-0.5 ديسيبل /11 ديسيبل"
        assert detect_document_type(text) == "router_signal_screen"

    def test_arabic_router_by_cell_label(self):
        text = "IMSI\n605030964873447\nخلايا\n083B80B/-"
        assert detect_document_type(text) == "router_signal_screen"

    def test_equipment_label_not_confused(self):
        text = "MAC: 98A942123456\nS/N: W51P740912\nModel: ZLT W51"
        assert detect_document_type(text) == "EQUIPMENT_LABEL"


# ---------------------------------------------------------------------------
# parse_router_screen — French variant (exact values)
# ---------------------------------------------------------------------------

class TestParseRouterFrench:
    def setup_method(self):
        self.result = parse_router_screen(FRENCH_OCR)

    def test_type(self):
        assert self.result["type"] == "router_signal_screen"

    def test_imsi(self):
        assert self.result["imsi"]["value"] == "605030968128782"

    def test_imei(self):
        assert self.result["imei"]["value"] == "861321072113879"

    def test_pci(self):
        assert self.result["pci"]["value"] == "487+487+487/876"

    def test_sinr(self):
        assert self.result["sinr"]["value"] == "3.5dB/15dB"

    def test_rsrp(self):
        assert self.result["rsrp"]["value"] == "-75dBm/-73dBm"

    def test_rssi(self):
        assert self.result["rssi"]["value"] == "-60dBm/-62dBm"

    def test_earfcn(self):
        assert self.result["earfcn"]["value"] == "101+1800+6400/628704"

    def test_rsrq(self):
        assert self.result["rsrq"]["value"] == "-14.0dB/-11dB"

    def test_cell_id(self):
        val = self.result["cellId"]["value"]
        assert val is not None and "075C415" in val.upper()

    def test_no_review_needed(self):
        assert self.result["needsReview"] is False

    def test_label_rule_source(self):
        assert self.result["imsi"]["source"] == "label_rule"
        assert self.result["sinr"]["source"] == "label_rule"

    def test_high_confidence_for_label_fields(self):
        for field in ("imsi", "imei", "sinr", "rsrp", "rssi", "rsrq"):
            assert self.result[field]["confidenceLevel"] == "high", (
                f"{field}: {self.result[field]}"
            )


# ---------------------------------------------------------------------------
# parse_router_screen — Arabic variant (exact values, user-specified)
# ---------------------------------------------------------------------------

class TestParseRouterArabicExact:
    """Exact value verification using the Arabic OCR rawText from the bug report."""

    def setup_method(self):
        self.result = parse_router_screen(ARABIC_OCR)

    def test_sinr_exact(self):
        assert self.result["sinr"]["value"] == "-0.5dB/11dB"

    def test_rsrp_exact(self):
        assert self.result["rsrp"]["value"] == "-87dBm/-112dBm"

    def test_rssi_exact(self):
        assert self.result["rssi"]["value"] == "-69dBm/-100dBm"

    def test_rsrq_exact(self):
        assert self.result["rsrq"]["value"] == "-16.0dB/-11dB"

    def test_earfcn_exact(self):
        assert self.result["earfcn"]["value"] == "1800+6400+101/628704"

    def test_cell_id_exact(self):
        assert self.result["cellId"]["value"] == "083B80B/-"

    def test_imsi(self):
        assert self.result["imsi"]["value"] == "605030964873447"

    def test_imei(self):
        assert self.result["imei"]["value"] == "861321070112543"

    def test_pci(self):
        assert self.result["pci"]["value"] == "147+147+147/364"

    def test_no_review_needed(self):
        assert self.result["needsReview"] is False

    def test_sinr_not_taken_from_earfcn(self):
        # Guard against regression where order fallback picks 1800+... or 628704
        val = self.result["sinr"]["value"]
        assert "1800" not in (val or "")
        assert "628704" not in (val or "")

    def test_rsrp_not_taken_from_cell_id(self):
        val = self.result["rsrp"]["value"]
        assert "083" not in (val or "") or "dBm" in (val or "")

    def test_sinr_source_is_scoped_fallback(self):
        assert self.result["sinr"]["source"] == "scoped_fallback"

    def test_rsrp_source_is_scoped_fallback(self):
        assert self.result["rsrp"]["source"] == "scoped_fallback"

    def test_scoped_fallback_is_medium_confidence(self):
        assert self.result["sinr"]["confidenceLevel"] == "medium"
        assert self.result["rsrp"]["confidenceLevel"] == "medium"


# ---------------------------------------------------------------------------
# parse_router_screen — multi-line RSRP (Arabic OCR splits value across lines)
# ---------------------------------------------------------------------------

class TestParseRouterMultilineRSRP:
    """Regression test for RSRP split across 4 lines in Arabic OCR output."""

    def setup_method(self):
        self.result = parse_router_screen(ARABIC_OCR_MULTILINE_RSRP)

    def test_rsrp_exact(self):
        assert self.result["rsrp"]["value"] == "-87dBm/-112dBm"

    def test_rsrp_stronger_signal_first(self):
        val = self.result["rsrp"]["value"]
        assert val is not None
        parts = val.replace("dBm", "").split("/")
        assert len(parts) == 2
        # -87 > -112 in dBm (closer to 0 = stronger), so -87 must come first
        assert float(parts[0]) > float(parts[1])

    def test_sinr_correct(self):
        assert self.result["sinr"]["value"] == "-0.5dB/11dB"

    def test_rssi_correct(self):
        assert self.result["rssi"]["value"] == "-69dBm/-100dBm"

    def test_rsrq_correct(self):
        assert self.result["rsrq"]["value"] == "-16.0dB/-11dB"

    def test_earfcn_correct(self):
        assert self.result["earfcn"]["value"] == "1800+6400+101/628704"

    def test_cell_id_correct(self):
        assert self.result["cellId"]["value"] == "083B80B/-"

    def test_no_review_needed(self):
        assert self.result["needsReview"] is False

    def test_rsrp_not_partial(self):
        # Guard: single-value result ("-112dBm") must not be returned
        val = self.result["rsrp"]["value"]
        assert val is not None
        assert "/" in val, f"Expected a pair, got: {val}"

    def test_rsrp_no_earfcn_numbers(self):
        val = self.result["rsrp"]["value"] or ""
        assert "1800" not in val
        assert "6400" not in val
        assert "628704" not in val

    def test_source_is_scoped_fallback(self):
        assert self.result["rsrp"]["source"] == "scoped_fallback"

    def test_confidence_is_medium(self):
        assert self.result["rsrp"]["confidenceLevel"] == "medium"


# ---------------------------------------------------------------------------
# IMS| label noise
# ---------------------------------------------------------------------------

class TestImsiLabelNoise:
    def test_imsi_extracted_despite_pipe_noise(self):
        result = parse_router_screen(ARABIC_OCR_IMSI_NOISE)
        assert result["imsi"]["value"] == "605030964873447"


# ---------------------------------------------------------------------------
# Confidence / null invariants
# ---------------------------------------------------------------------------

class TestConfidenceInvariants:
    def test_null_value_gives_zero_confidence(self):
        fr = field_result(None, 0.95, "some_source")
        assert fr["confidence"] == 0.0

    def test_null_value_gives_low_confidence_level(self):
        fr = field_result(None, 0.95, "some_source")
        assert fr["confidenceLevel"] == "low"

    def test_non_null_value_keeps_confidence(self):
        fr = field_result("hello", 0.92, "label_rule")
        assert fr["confidence"] == 0.92
        assert fr["confidenceLevel"] == "high"

    def test_router_missing_field_triggers_review(self):
        result = parse_router_screen("IMSI\n605030968128782")
        assert result["needsReview"] is True

    def test_router_complete_french_no_review(self):
        result = parse_router_screen(FRENCH_OCR)
        assert result["needsReview"] is False

    def test_router_complete_arabic_no_review(self):
        result = parse_router_screen(ARABIC_OCR)
        assert result["needsReview"] is False


# ---------------------------------------------------------------------------
# Equipment label with corrupted OCR text (Arabic+English reader artefacts)
# ---------------------------------------------------------------------------

EQUIPMENT_LABEL_OCR = """\
5G Wireless Router
Model: ZLT X16
|P: 192.168.0. 1
TOZ٤٥
|ME|:860250060329216
MAC:98A94ZFFSEE9
RoHS
$/N:X16860250060329216
Made in China"""


class TestParseEquipmentLabelCorrupted:
    """Parse an equipment label whose OCR text has |ME|, Z in MAC, and $/N."""

    def setup_method(self):
        self.result = parse_equipment_label(EQUIPMENT_LABEL_OCR)

    def test_detected_as_equipment_label(self):
        assert detect_document_type(EQUIPMENT_LABEL_OCR) == "EQUIPMENT_LABEL"

    def test_imei(self):
        assert self.result["imei"]["value"] == "860250060329216"

    def test_mac_address(self):
        assert self.result["macAddress"]["value"] == "98A942FF5EE9"

    def test_serial_number(self):
        assert self.result["serialNumber"]["value"] == "X16860250060329216"
