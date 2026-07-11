import re
from typing import Any, Dict, List, Optional, Tuple


ROUTER_FIELDS = {
    "imsi": ["imsi", "1msi", "lmsi", "ims|", "ims1"],
    "imei": [
        "imei", "1mei", "lmei",
        "imei رقم", "رقم imei",
    ],
    "pci": ["pci", "pcl"],
    "sinr": [
        "sinr", "snr",
        # Normalized Arabic (ة→ه, إ→ا, ى→ي): نسبة الإشارة إلى → نسبه الاشاره الي
        "نسبه الاشاره الي الضوضاء والتداخل",
        "نسبه الاشاره",
        "نسبة الاشارة",
    ],
    "rsrp": [
        "rsrp", "r5rp",
        "سعر التحجره المقترح",
        "سعر التحجرة المقترح",
    ],
    "rssi": ["rssi", "rss1", "rssl"],
    "earfcn": ["earfcn", "earfon", "earfcm", "earf"],
    "rsrq": ["rsrq", "r5rq"],
    "cell_id": [
        "cell id", "cellid", "cell",
        "خلايا ايس", "خلايا إيس", "خلايا",
    ],
    "band": ["bande actuelle", "bande", "current band", "band actuelle"],
}

KNOWN_FIELD_ORDER = [
    "imsi", "imei", "pci", "sinr", "rsrp", "rssi",
    "earfcn", "rsrq", "cell_id", "band",
]

FIELD_UNITS = {
    "sinr": "dB",
    "rsrp": "dBm",
    "rssi": "dBm",
    "rsrq": "dB",
}

CRITICAL_ROUTER_FIELDS = {"imsi", "imei", "pci", "sinr", "rsrp", "rssi", "earfcn", "rsrq"}

# Keys of the parsed router result dict, in output order.
# Used by the per-field merge of the EN and AR+EN OCR passes.
ROUTER_RESULT_FIELDS = [
    "imsi", "imei", "pci", "sinr", "rsrp", "rssi",
    "earfcn", "rsrq", "cellId", "bandActuelle",
]

# Plausibility ranges per signal metric. Values outside these bounds are
# physically impossible on LTE equipment, so they can only be OCR misreads
# (e.g. the unit digit of a corrupted "d8" token grabbed as a value).
METRIC_PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "sinr": (-20.0, 40.0),    # dB
    "rsrp": (-140.0, -40.0),  # dBm
    "rssi": (-120.0, -10.0),  # dBm
    "rsrq": (-30.0, 0.0),     # dB — RSRQ is never positive
}

SPEEDTEST_DOWNLOAD_LABELS = [
    "descendant",
    "download",
    "downlink",
    "download speed",
]

SPEEDTEST_UPLOAD_LABELS = [
    "ascendant",
    "upload",
    "uplink",
    "upload speed",
]


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: digit variants, alef forms, ya, ta marbuta."""
    for ar_d, d in zip("٠١٢٣٤٥٦٧٨٩", "0123456789"):
        text = text.replace(ar_d, d)
    for ar_d, d in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"):
        text = text.replace(ar_d, d)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text


# Compiled once: dBm patterns must come before plain dB to avoid partial replacement.
_ARABIC_DBM_RE = re.compile(
    r"ديسيبل\s+(?:م[يل]+\s*)?واط"   # ديسيبل واط / ديسيبل ملي واط / ديسيبل ميلي واط
    r"|م[يل]+\s*واط",                # ملي واط / ميلي واط  (standalone suffix)
    re.UNICODE,
)
_ARABIC_DB_RE = re.compile(r"ديسيبل", re.UNICODE)


def normalize_arabic_units(text: str) -> str:
    """Replace Arabic dBm/dB words with standard ASCII abbreviations."""
    text = _ARABIC_DBM_RE.sub("dBm", text)
    text = _ARABIC_DB_RE.sub("dB", text)
    return text


def normalize_text(text: str) -> str:
    text = normalize_arabic(text)
    text = text.replace(",", ".")
    text = text.replace("،", ".")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_lines(text: str) -> List[str]:
    return [line.strip() for line in normalize_text(text).splitlines() if line.strip()]


def fix_ocr_noise(value: str) -> str:
    value = value.strip()

    value = value.replace("~", "-")
    value = value.replace("O", "0")
    value = value.replace("o", "0")
    value = value.replace("I", "1")
    value = value.replace("l", "1")

    value = re.sub(r"S(?=\d|S|dB|dBm)", "5", value)

    return value


def normalize_db_noise(value: str) -> str:
    """Repair dB/dBm unit tokens corrupted by OCR.

    The AR+EN recognition model misreads B as 8/3/O on Latin text
    ("3.5dB/15dB" -> "3. 5d3/1 50B"). Order matters: rejoin decimals first,
    then repair unit tokens, then merge digits split right before an intact
    dB token.
    """
    # "3. 5" -> "3.5": decimal point split by a stray space
    value = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", value)
    # "d8m"/"d3m"/"dOm"/"d0m" -> "dBm" (before the dB rule so the m is kept)
    value = re.sub(r"(?<=\d)\s*d[830oO]m\b", "dBm", value)
    # "d8"/"d3"/"dO"/"d0" -> "dB"
    value = re.sub(r"(?<=\d)\s*d[830oO]\b", "dB", value)
    # digit + "0B"/"OB" -> digit + "dB": here the d itself was misread as 0/O
    value = re.sub(r"(?<=\d)[0O]Bm\b", "dBm", value)
    value = re.sub(r"(?<=\d)[0O]B\b", "dB", value)
    # "1 5dB" -> "15dB": digits split by a stray space right before the unit
    value = re.sub(r"(\d)\s+(\d+(?:\.\d+)?\s*dBm?\b)", r"\1\2", value)
    return value


def metric_pair_in_range(metric_type: str, pair: Optional[str]) -> Optional[bool]:
    """Check every number in an extracted pair against the metric's plausible range.

    Returns None when no range is defined for the metric or the pair is empty,
    True/False otherwise.
    """
    rng = METRIC_PLAUSIBLE_RANGES.get(metric_type)
    if rng is None or not pair:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", pair)
    if not nums:
        return False
    lo, hi = rng
    return all(lo <= float(n) <= hi for n in nums)


def detect_document_type(text: str) -> str:
    low = normalize_text(text).lower()

    # Equipment label: MAC + serial number together are unambiguous.
    # Checked first because equipment labels can also contain "imei".
    has_mac = bool(re.search(r"\bmac\b", low))
    has_sn = bool(re.search(r"\bs/n\b|\bserial\b|\$\s*/\s*n\b", low))
    has_label_brand = "tozed" in low or "zlt" in low or "wireless router" in low

    if (has_mac and has_sn) or (has_label_brand and (has_mac or has_sn)):
        return "EQUIPMENT_LABEL"

    has_router_latin = (
        "imsi" in low or "imei" in low
        or "rsrp" in low or "rsrq" in low
        or "earfcn" in low or "sinr" in low
    )
    has_router_arabic = (
        "رقم" in low        # IMEI رقم
        or "نسبه" in low    # normalized SINR label (ة→ه)
        or "نسبة" in low
        or "خلايا" in low   # Cell ID
        or "ديسيبل" in low  # Arabic dB/dBm unit word
    )

    if has_router_latin or has_router_arabic:
        return "router_signal_screen"

    if (
        "speedtest" in low
        or "descendant" in low
        or "ascendant" in low
        or "download" in low
        or "upload" in low
    ):
        return "speedtest_screen"

    return "unknown"


def calculate_bbox_center(bbox) -> Tuple[float, float]:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def line_contains_label(line: str, labels: List[str]) -> bool:
    low = line.lower().strip()

    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", low):
            return True

    return False


def extract_first_number(text: str) -> Optional[str]:
    text = fix_ocr_noise(text)
    match = re.search(r"\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def parse_float(value: str) -> Optional[float]:
    value = fix_ocr_noise(value)
    value = re.sub(r"[^0-9.\-]", "", value)

    try:
        return float(value)
    except ValueError:
        return None


def confidence_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def field_result(value: Any, confidence: float, source: str) -> Dict[str, Any]:
    if value is None:
        confidence = 0.0
    return {
        "value": value,
        "confidence": round(confidence, 4),
        "confidenceLevel": confidence_level(confidence),
        "source": source,
    }


def _ocr_item_confidence(
    raw_value: Optional[str],
    ocr_items: Optional[List[Dict[str, Any]]],
) -> Optional[float]:
    """OCR confidence of the item a field value came from.

    Matches the normalized raw value line against normalized item texts:
    exact match preferred, otherwise containment (>= 4 chars, longest match
    wins). Returns None when no item matches — the caller then keeps its
    rule-based score.
    """
    if not raw_value or not ocr_items:
        return None
    target = fix_ocr_noise(normalize_text(str(raw_value))).lower()
    if not target:
        return None

    exact: Optional[float] = None
    partial: List[Tuple[int, float]] = []
    for item in ocr_items:
        if item.get("source") == "barcode":
            continue
        text = fix_ocr_noise(normalize_text(item.get("text", ""))).lower()
        if not text:
            continue
        conf = float(item.get("confidence", 0.0))
        if text == target:
            exact = conf if exact is None else max(exact, conf)
        elif len(text) >= 4 and (text in target or target in text):
            partial.append((len(text), conf))

    if exact is not None:
        return exact
    if partial:
        return max(partial)[1]
    return None


# -----------------------
# ROUTER PARSING
# -----------------------

def _find_label_value(
    lines: List[str], aliases: List[str]
) -> Tuple[Optional[str], bool]:
    """Search normalized lines for a label alias; return the following value line."""
    for i, line in enumerate(lines):
        norm = normalize_text(line).lower()
        for alias in aliases:
            if alias in norm:
                # Inline value: text after the alias on the same line
                rest = norm[norm.find(alias) + len(alias):].strip().lstrip(": \t")
                if rest and re.search(r"\d", rest):
                    return fix_ocr_noise(rest), True
                # Next line
                if i + 1 < len(lines):
                    return fix_ocr_noise(normalize_text(lines[i + 1])), True
                return None, True
    return None, False


def _is_value_line(line: str) -> bool:
    """Return True if a line looks like a field value rather than a label."""
    norm = normalize_text(line)
    norm_u = normalize_arabic_units(norm)
    if re.search(r"\d{8,}", norm):
        return True
    if re.search(r"-?\d+(?:\.\d+)?\s*dBm?", norm_u, re.IGNORECASE):
        return True
    if re.search(r"\d+[+/]\d+", norm):
        return True
    if re.search(r"[0-9A-Fa-f]{5,}", norm) and re.search(r"[A-Fa-f]", norm):
        return True
    return False


def _order_based_fallback(lines: List[str]) -> Dict[str, Optional[str]]:
    """Map value-looking lines to the known fixed field order."""
    value_lines = [
        fix_ocr_noise(normalize_text(line)) for line in lines if _is_value_line(line)
    ]
    return {
        field: value_lines[i] if i < len(value_lines) else None
        for i, field in enumerate(KNOWN_FIELD_ORDER)
    }


def extract_imsi_imei(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = fix_ocr_noise(normalize_text(value))
    match = re.search(r"\d{10,20}", value)
    return match.group(0) if match else None


def extract_pci(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = fix_ocr_noise(normalize_text(value))
    match = re.search(r"\d+(?:[+/]\d+)+", value)
    if match:
        return match.group(0)
    match = re.search(r"\d+", value)
    return match.group(0) if match else None


def extract_earfcn(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = fix_ocr_noise(normalize_text(value))
    match = re.search(r"\d+(?:[+]\d+)+(?:[/]\d+)?|\d+(?:[/]\d+)+", value)
    if match:
        return match.group(0)
    match = re.search(r"\d+", value)
    return match.group(0) if match else None


def extract_cell_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = normalize_text(value).strip()
    match = re.search(r"[0-9A-Fa-f]{5,}(?:/[-*])?", value, re.IGNORECASE)
    if match:
        return match.group(0)
    match = re.search(r"[0-9A-Fa-f]{3,}", value, re.IGNORECASE)
    return match.group(0) if match else None


def extract_metric_pair(value: Optional[str], unit: str = "dB") -> Optional[str]:
    """Extract a dB/dBm pair from French or Arabic OCR text.

    Handles formats:
      3.5dB/15dB, -75dBm/-73dBm           (standard)
      -0.5 ديسيبل /11 ديسيبل              (Arabic dB)
      -87 ديسيبل واط / -112 ديسيبل واط    (Arabic dBm)
    """
    if not value:
        return None
    value = normalize_db_noise(normalize_text(value))
    value = normalize_arabic_units(fix_ocr_noise(value))

    # Standard: -75dBm/-73dBm or 3.5dB/15dB
    m = re.search(
        r"(-?\d+(?:\.\d+)?)\s*dBm?\s*/\s*(-?\d+(?:\.\d+)?)\s*dBm?",
        value,
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)}{unit}/{m.group(2)}{unit}"

    # Arabic/mixed: split on slash, pick last number left and first number right
    if "/" in value:
        parts = value.split("/", 1)
        left_nums = re.findall(r"-?\d+(?:\.\d+)?", parts[0])
        right_nums = re.findall(r"-?\d+(?:\.\d+)?", parts[1])
        if left_nums and right_nums:
            return f"{left_nums[-1]}{unit}/{right_nums[0]}{unit}"
        all_nums = left_nums + right_nums
        if len(all_nums) >= 2:
            return f"{all_nums[0]}{unit}/{all_nums[1]}{unit}"
        if len(all_nums) == 1:
            return f"{all_nums[0]}{unit}"

    # No slash: take first two numbers found
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(nums) >= 2:
        return f"{nums[0]}{unit}/{nums[1]}{unit}"
    if len(nums) == 1:
        return f"{nums[0]}{unit}"
    return None


def normalize_signal_metric(line: str, metric_type: str) -> Optional[str]:
    """Extract a signed-decimal pair from a signal line and format with correct unit.

    Normalizes Arabic unit words before pattern matching so both
    '-0.5 ديسيبل /11 ديسيبل' and '3.5dB/15dB' produce the same code path.
    """
    unit = FIELD_UNITS.get(metric_type, "dB")
    return extract_metric_pair(line, unit)


def _find_sinr_rsrp_scoped(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Scoped fallback: locate SINR and RSRP between the PCI and RSSI labels.

    Arabic OCR labels for these fields are often corrupted. We locate the block
    by its reliable Latin boundaries (PCI / RSSI) instead.

    SINR  — the first line inside the block that carries a signal unit and a number.
    RSRP  — collected from the next 3-5 lines after the SINR value line (Arabic OCR
            sometimes splits one value across several lines: unit word on its own line,
            value on the next). All signed numbers in the window are extracted, filtered
            to the realistic RSRP range [-140, -40] dBm, sorted descending so the
            stronger signal comes first, then formatted as "XdBm/YdBm".
    """
    pci_idx: Optional[int] = None
    rssi_idx: Optional[int] = None

    for i, line in enumerate(lines):
        norm = normalize_text(line).lower()
        if pci_idx is None:
            for alias in ROUTER_FIELDS["pci"]:
                if alias in norm:
                    pci_idx = i
                    break
        if rssi_idx is None:
            for alias in ROUTER_FIELDS["rssi"]:
                if alias in norm:
                    rssi_idx = i
                    break

    if pci_idx is None or rssi_idx is None or rssi_idx <= pci_idx + 1:
        return None, None

    block = lines[pci_idx + 1 : rssi_idx]

    # SINR: first line in the block that has a signal unit *and* a number.
    sinr_line: Optional[str] = None
    sinr_block_idx: Optional[int] = None
    for i, line in enumerate(block):
        norm_u = normalize_arabic_units(normalize_text(line))
        if re.search(r"dBm?", norm_u, re.IGNORECASE) and re.search(r"-?\d", norm_u):
            sinr_line = line
            sinr_block_idx = i
            break

    # RSRP: join the next 3-5 lines after the SINR value and harvest all numbers
    # in the RSRP signal range [-140, -40] dBm.  Sort descending (stronger first).
    rsrp_value: Optional[str] = None
    if sinr_block_idx is not None:
        rsrp_start = sinr_block_idx + 1
        rsrp_end = min(rsrp_start + 5, len(block))
        rsrp_text = " ".join(block[rsrp_start:rsrp_end])
        rsrp_norm = normalize_arabic_units(normalize_text(rsrp_text))

        raw_nums = re.findall(r"-?\d+(?:\.\d+)?", rsrp_norm)
        valid = sorted(
            [float(n) for n in raw_nums if -140.0 <= float(n) <= -40.0],
            reverse=True,
        )

        def _fmt(n: float) -> str:
            return str(int(n)) if n == int(n) else str(n)

        if len(valid) >= 2:
            rsrp_value = f"{_fmt(valid[0])}dBm/{_fmt(valid[1])}dBm"
        elif len(valid) == 1:
            rsrp_value = f"{_fmt(valid[0])}dBm"

    return sinr_line, rsrp_value


def parse_router_screen(
    text: str,
    ocr_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    lines = split_lines(text)

    raw: Dict[str, Optional[str]] = {}
    sources: Dict[str, str] = {}

    # Pass 1: label-based extraction
    for field, aliases in ROUTER_FIELDS.items():
        value_line, found = _find_label_value(lines, aliases)
        if found and value_line:
            raw[field] = value_line
            sources[field] = "label_rule"

    # Pass 2: nearby-line (two lines after the label)
    for field, aliases in ROUTER_FIELDS.items():
        if raw.get(field):
            continue
        for i, line in enumerate(lines):
            norm = normalize_text(line).lower()
            matched = False
            for alias in aliases:
                if alias in norm:
                    matched = True
                    break
            if matched and i + 2 < len(lines):
                candidate = fix_ocr_noise(normalize_text(lines[i + 2]))
                if re.search(r"\d", candidate):
                    raw[field] = candidate
                    sources[field] = "nearby_line"

    # Pass 3a: scoped block fallback for SINR / RSRP only.
    # Arabic labels for these two fields are often corrupted beyond alias matching.
    # We constrain the search to the block between the PCI and RSSI labels so we
    # never accidentally pull EARFCN or Cell ID lines.
    if not raw.get("sinr") or not raw.get("rsrp"):
        sinr_line, rsrp_line = _find_sinr_rsrp_scoped(lines)
        if not raw.get("sinr") and sinr_line:
            raw["sinr"] = sinr_line
            sources["sinr"] = "scoped_fallback"
        if not raw.get("rsrp") and rsrp_line:
            raw["rsrp"] = rsrp_line
            sources["rsrp"] = "scoped_fallback"

    # Pass 3b: order-based fallback for remaining missing fields.
    # SINR and RSRP are excluded — they must come from a label or scoped block,
    # never from a global position match that might hit EARFCN / Cell ID.
    _ORDER_FALLBACK_EXCLUDE = {"sinr", "rsrp"}
    missing = [
        f for f in KNOWN_FIELD_ORDER
        if not raw.get(f) and f not in _ORDER_FALLBACK_EXCLUDE
    ]
    if missing:
        order_values = _order_based_fallback(lines)
        for field in missing:
            if order_values.get(field):
                raw[field] = order_values[field]
                sources[field] = "order_fallback"

    def _conf(field: str, value: Any) -> Tuple[float, str]:
        if value is None:
            return 0.0, sources.get(field, "not_found")
        src = sources.get(field, "unknown")
        if src == "label_rule":
            return 0.92, src
        if src == "nearby_line":
            return 0.75, src
        if src == "scoped_fallback":
            return 0.70, src
        if src == "order_fallback":
            return 0.60, src
        return 0.70, src

    def _make(field: str, extractor, raw_val: Optional[str]) -> Dict[str, Any]:
        extracted = extractor(raw_val) if raw_val else None
        conf, src = _conf(field, extracted)
        if extracted is not None:
            # Honest confidence: a label_rule match means nothing if the OCR
            # item carrying the value was itself low-confidence.
            item_conf = _ocr_item_confidence(raw_val, ocr_items)
            if item_conf is not None:
                conf = min(conf, item_conf)
        return field_result(extracted, conf, src)

    extracted: Dict[str, Any] = {
        "type": "router_signal_screen",
        "imsi":        _make("imsi",    extract_imsi_imei,                                  raw.get("imsi")),
        "imei":        _make("imei",    extract_imsi_imei,                                  raw.get("imei")),
        "pci":         _make("pci",     extract_pci,                                        raw.get("pci")),
        "sinr":        _make("sinr",    lambda v: normalize_signal_metric(v, "sinr"),       raw.get("sinr")),
        "rsrp":        _make("rsrp",    lambda v: normalize_signal_metric(v, "rsrp"),       raw.get("rsrp")),
        "rssi":        _make("rssi",    lambda v: normalize_signal_metric(v, "rssi"),       raw.get("rssi")),
        "earfcn":      _make("earfcn",  extract_earfcn,                                     raw.get("earfcn")),
        "rsrq":        _make("rsrq",    lambda v: normalize_signal_metric(v, "rsrq"),       raw.get("rsrq")),
        "cellId":      _make("cell_id", extract_cell_id,                                    raw.get("cell_id")),
        "bandActuelle": _make("band",   lambda v: v.strip() if v else None,                 raw.get("band")),
    }

    # Range validation: an out-of-range pair can only be an OCR misread.
    for metric in METRIC_PLAUSIBLE_RANGES:
        entry = extracted[metric]
        in_range = metric_pair_in_range(metric, entry["value"])
        if in_range is None:
            continue
        entry["rangeValid"] = in_range
        if not in_range:
            entry["confidence"] = round(min(entry["confidence"], 0.4), 4)
            entry["confidenceLevel"] = confidence_level(entry["confidence"])

    extracted["needsReview"] = router_needs_review(extracted)

    return extracted


# -----------------------
# SPEEDTEST TEXT PARSING
# -----------------------

def parse_speedtest_text_rules(text: str) -> Dict[str, Any]:
    lines = split_lines(text)

    descendant_index = None
    ascendant_index = None

    for i, line in enumerate(lines):
        if descendant_index is None and line_contains_label(line, SPEEDTEST_DOWNLOAD_LABELS):
            descendant_index = i

        if ascendant_index is None and line_contains_label(line, SPEEDTEST_UPLOAD_LABELS):
            ascendant_index = i

    values = []

    if descendant_index is not None and ascendant_index is not None:
        start = max(descendant_index, ascendant_index)

        for i in range(start + 1, len(lines)):
            line = lines[i]

            if "mbps" in line.lower():
                break

            num = extract_first_number(line)

            if num:
                val = parse_float(num)

                if val is not None and 1 < val < 1000:
                    values.append(val)

    if len(values) >= 2:
        return {
            "descendant_mbps": field_result(values[0], 0.90, "text_layout_rule"),
            "ascendant_mbps": field_result(values[1], 0.90, "text_layout_rule"),
        }

    return {
        "descendant_mbps": field_result(None, 0.0, "text_layout_rule_failed"),
        "ascendant_mbps": field_result(None, 0.0, "text_layout_rule_failed"),
    }


# -----------------------
# SPEEDTEST LAYOUT PARSING
# -----------------------

def parse_speedtest_layout_rules(ocr_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ocr_items:
        return {
            "descendant_mbps": field_result(None, 0.0, "layout_rule_failed"),
            "ascendant_mbps": field_result(None, 0.0, "layout_rule_failed"),
        }

    enriched = []

    for item in ocr_items:
        x, y = calculate_bbox_center(item["bbox"])
        enriched.append(
            {
                **item,
                "x": x,
                "y": y,
                "low": item["text"].lower().strip(),
            }
        )

    download_labels = [
        item for item in enriched
        if line_contains_label(item["text"], SPEEDTEST_DOWNLOAD_LABELS)
    ]

    upload_labels = [
        item for item in enriched
        if line_contains_label(item["text"], SPEEDTEST_UPLOAD_LABELS)
    ]

    numbers = []

    for item in enriched:
        num = extract_first_number(item["text"])

        if num:
            val = parse_float(num)

            if val is not None and 1 < val < 1000:
                if "ms" not in item["low"] and "ping" not in item["low"] and "gigue" not in item["low"]:
                    numbers.append({**item, "value": val})

    if not download_labels or not upload_labels:
        return {
            "descendant_mbps": field_result(None, 0.0, "layout_rule_no_labels"),
            "ascendant_mbps": field_result(None, 0.0, "layout_rule_no_labels"),
        }

    download_label = download_labels[0]
    upload_label = upload_labels[0]

    download_candidates = []
    upload_candidates = []

    for number in numbers:
        if number["y"] <= min(download_label["y"], upload_label["y"]):
            continue

        vertical_distance_download = abs(number["y"] - download_label["y"])
        horizontal_distance_download = abs(number["x"] - download_label["x"])

        vertical_distance_upload = abs(number["y"] - upload_label["y"])
        horizontal_distance_upload = abs(number["x"] - upload_label["x"])

        if vertical_distance_download < 250:
            download_candidates.append(
                (
                    horizontal_distance_download + vertical_distance_download * 0.35,
                    number,
                )
            )

        if vertical_distance_upload < 250:
            upload_candidates.append(
                (
                    horizontal_distance_upload + vertical_distance_upload * 0.35,
                    number,
                )
            )

    download_candidates.sort(key=lambda x: x[0])
    upload_candidates.sort(key=lambda x: x[0])

    descendant = download_candidates[0][1] if download_candidates else None
    ascendant = upload_candidates[0][1] if upload_candidates else None

    return {
        "descendant_mbps": field_result(
            descendant["value"] if descendant else None,
            min(descendant["confidence"], 0.95) if descendant else 0.0,
            "bbox_layout_rule",
        ),
        "ascendant_mbps": field_result(
            ascendant["value"] if ascendant else None,
            min(ascendant["confidence"], 0.95) if ascendant else 0.0,
            "bbox_layout_rule",
        ),
    }


# -----------------------
# ML-LIKE PARSING
# -----------------------

def ml_like_speedtest_parser(text: str) -> Dict[str, Any]:
    """
    Lightweight ML-like heuristic parser.
    Not a real trained model, but behaves smarter than raw regex:
    - understands nearby labels
    - handles FR/EN labels
    - handles stacked OCR order
    """

    lines = split_lines(text)

    useful_numbers = []

    for i, line in enumerate(lines):
        low = line.lower()

        noise = [
            "ping",
            "gigue",
            "jitter",
            "ms",
            "identifier",
            "identifiant",
            "test id",
            "resultat",
            "detail",
            "downdetector",
            "carte",
            "map",
            "sousse",
            "ooredoo",
            "publicite",
            "ads",
        ]

        if any(word in low for word in noise):
            continue

        num = extract_first_number(line)

        if num:
            val = parse_float(num)

            if val is not None and 1 < val < 1000:
                useful_numbers.append((i, val))

    has_download = any(line_contains_label(line, SPEEDTEST_DOWNLOAD_LABELS) for line in lines)
    has_upload = any(line_contains_label(line, SPEEDTEST_UPLOAD_LABELS) for line in lines)

    if has_download and has_upload and len(useful_numbers) >= 2:
        return {
            "descendant_mbps": field_result(useful_numbers[0][1], 0.75, "ml_like_parser"),
            "ascendant_mbps": field_result(useful_numbers[1][1], 0.75, "ml_like_parser"),
        }

    return {
        "descendant_mbps": field_result(None, 0.0, "ml_like_parser_failed"),
        "ascendant_mbps": field_result(None, 0.0, "ml_like_parser_failed"),
    }


def choose_best_speedtest_result(
    text_result: Dict[str, Any],
    layout_result: Dict[str, Any],
    ml_result: Dict[str, Any],
) -> Dict[str, Any]:

    # 1) Prefer text layout rule when it found two different decimal Mbps values
    d = text_result["descendant_mbps"]["value"]
    u = text_result["ascendant_mbps"]["value"]

    if d is not None and u is not None and d != u:
        if isinstance(d, float) and isinstance(u, float):
            return text_result

    # 2) Then ML-like parser
    d = ml_result["descendant_mbps"]["value"]
    u = ml_result["ascendant_mbps"]["value"]

    if d is not None and u is not None and d != u:
        return ml_result

    # 3) Last option: bbox layout
    d = layout_result["descendant_mbps"]["value"]
    u = layout_result["ascendant_mbps"]["value"]

    if d is not None and u is not None and d != u:
        return layout_result

    return text_result


def parse_speedtest_screen(text: str, ocr_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    text_result = parse_speedtest_text_rules(text)
    layout_result = parse_speedtest_layout_rules(ocr_items or [])
    ml_result = ml_like_speedtest_parser(text)

    best = choose_best_speedtest_result(text_result, layout_result, ml_result)

    result = {
        "type": "speedtest_screen",
        "descendant_mbps": best["descendant_mbps"],
        "ascendant_mbps": best["ascendant_mbps"],
    }

    result["needsReview"] = needs_review(result)

    return result


# -----------------------
# EQUIPMENT LABEL CONSTANTS
# -----------------------

# Known OUI prefixes for ZTE/TOZED routers commonly deployed in Tunisia.
# If the first 3 octets of a parsed MAC match a known OUI, we can boost
# confidence — OCR didn't invent the address, it's from a real device family.
KNOWN_ROUTER_OUIS: set = {
    "98A942",  # ZTE Corporation (most common on ZLT X16/W51)
    "0019CB",  # ZTE Corporation
    "001D0F",  # ZTE Corporation
    "645106",  # ZTE Corporation
    "A46006",  # ZTE Corporation
    "F8A863",  # ZTE Corporation
    "C864C7",  # ZTE Corporation
}

# Character pairs EasyOCR commonly confuses in hex context.
# Used ONLY for OUI-prefix correction — never applied to the full MAC blindly.
# 4↔A and 6↔8 are the reported failures; B↔8 is included as it shares the pattern.
_OUI_CONFUSION_PAIRS: List[Tuple[str, str]] = [
    ("4", "A"),
    ("A", "4"),
    ("8", "B"),
    ("B", "8"),
    ("6", "8"),
    ("8", "6"),
]


def generate_oui_corrected_mac(mac: str) -> Optional[str]:
    """Try single-char substitutions in the OUI (first 6 chars) to match a known prefix.

    This is how the A/4 confusion is resolved: if OCR reads '984942' but
    '98A942' is a known ZTE OUI, we propose the corrected MAC as a candidate.
    Correction is limited to ONE character change to avoid combinatorial guessing.
    Returns the corrected 12-char MAC, or None if no known OUI can be reached.
    """
    oui = mac[:6].upper()
    suffix = mac[6:]

    if oui in KNOWN_ROUTER_OUIS:
        return None  # already valid, no correction needed

    for i in range(6):
        ch = oui[i]
        for orig, repl in _OUI_CONFUSION_PAIRS:
            if ch == orig:
                candidate_oui = oui[:i] + repl + oui[i + 1:]
                if candidate_oui in KNOWN_ROUTER_OUIS:
                    return candidate_oui + suffix

    return None


# -----------------------
# EQUIPMENT LABEL HELPERS
# -----------------------

def normalize_mac(value: str) -> Optional[str]:
    cleaned = re.sub(r"[:\-\s]", "", value).upper()
    if re.fullmatch(r"[0-9A-F]{12}", cleaned):
        return cleaned
    return None


def clean_model(value: str) -> Optional[str]:
    # Stop at double-space or tab (other fields that may follow on the same logical line)
    value = re.split(r"\s{2,}|\t", value)[0].strip()
    value = re.sub(r"[^A-Za-z0-9 \-_.]", "", value).strip()
    return value if value else None


def extract_first_match(
    patterns: List[str],
    text: str,
    flags: int = re.IGNORECASE,
) -> Optional[Any]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match
    return None


def fix_hex_noise(value: str) -> str:
    """Fix unambiguous OCR misreads in hexadecimal strings.

    Corrects characters never valid in hex (O, I, L, S, Z) to their numeric
    equivalents. Z→2 catches a common Arabic+English OCR misread. Does NOT
    touch 4↔A or 8↔B which are ambiguous.
    """
    v = value.upper()
    v = v.replace("O", "0")
    v = v.replace("I", "1")
    v = v.replace("L", "1")
    v = v.replace("S", "5")
    v = v.replace("Z", "2")
    return v


def extract_by_bbox_label(
    ocr_items: List[Dict[str, Any]],
    label_patterns: List[str],
    value_validator,
    same_row_y_tol: int = 35,
    below_y_max: int = 90,
) -> Optional[Dict[str, Any]]:
    """Find a value item positioned immediately right of or below a label item.

    More robust than flat-text regex when OCR returns items out of line order.
    Uses the same bbox-center approach as the speedtest layout parser.
    """
    regular = [it for it in ocr_items if it.get("source") not in ("barcode", "allowlist_hex") and it.get("bbox")]
    if not regular:
        return None

    # Find the label item
    label_item = None
    for item in regular:
        for pat in label_patterns:
            if re.search(pat, item["text"], re.IGNORECASE):
                label_item = item
                break
        if label_item:
            break
    if not label_item:
        return None

    lx, ly = calculate_bbox_center(label_item["bbox"])

    # Collect candidate value items: same row (right) or next row (below)
    matches: List[Tuple[float, Dict[str, Any], str]] = []
    for item in regular:
        if item is label_item:
            continue
        cx, cy = calculate_bbox_center(item["bbox"])
        dx, dy = cx - lx, cy - ly

        if abs(dy) <= same_row_y_tol and dx > 10:
            # Same row, to the right — prefer by proximity
            val = value_validator(item["text"])
            if val:
                matches.append((dx, item, val))

        elif 0 < dy <= below_y_max and abs(dx) < 200:
            # Next row, roughly aligned — weighted by vertical + horizontal distance
            val = value_validator(item["text"])
            if val:
                matches.append((dy * 1.5 + abs(dx), item, val))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    _, best_item, best_val = matches[0]
    return {
        "value": best_val,
        "text": best_item["text"],
        "confidence": best_item["confidence"],
    }


def collect_mac_candidates(
    block: str,
    ocr_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect all MAC candidates from text, allowlist-hex pass, and barcode.

    Separate dedup per source so OCR and barcode finding the same value are
    both retained — _resolve_candidate needs both to emit "ocr_and_barcode".
    fix_hex_noise is applied before normalize_mac to correct O→0, I→1, S→5.
    """
    candidates: List[Dict[str, Any]] = []
    ocr_seen: set = set()
    bc_seen: set = set()

    # A: text-rule matches with hex noise correction.
    # The capture group accepts O, I, L, S alongside real hex chars so that
    # fix_hex_noise can correct them before normalize_mac validates the result.
    _HEX_NOISY = r"[0-9A-Fa-fOILSZoilsz]"
    for match in re.finditer(
        rf"\bmac\s*:?\s*((?:{_HEX_NOISY}{{2}}[:\-]){{5}}{_HEX_NOISY}{{2}}|{_HEX_NOISY}{{12}})",
        block,
        re.IGNORECASE,
    ):
        norm = normalize_mac(fix_hex_noise(match.group(1)))
        if norm and norm not in ocr_seen:
            ocr_seen.add(norm)
            candidates.append({"value": norm, "raw": match.group(1), "source": "ocr_text_rule"})

    # Barcode items
    for item in ocr_items:
        if item.get("source") == "barcode" and item.get("barcode_type") == "mac":
            norm = normalize_mac(item["text"])
            if norm and norm not in bc_seen:
                bc_seen.add(norm)
                candidates.append({"value": norm, "raw": item["text"], "source": "barcode"})

    # OUI correction: for each OCR candidate whose first 3 bytes don't match a
    # known ZTE/TOZED OUI, try single-char confusion-pair substitutions (A↔4,
    # 6↔8, B↔8) in the OUI prefix only.  This is the targeted fix for the
    # "A read as 4" and "6 read as 8" problems without touching unvalidatable
    # parts of the address.
    all_seen = ocr_seen | bc_seen
    for cand in list(candidates):
        if cand.get("source") not in ("ocr_text_rule", "ocr_bbox_rule"):
            continue
        corrected = generate_oui_corrected_mac(cand["value"])
        if corrected and corrected not in all_seen:
            all_seen.add(corrected)
            candidates.append({
                "value": corrected,
                "raw": cand["value"],
                "original_ocr": cand["value"],
                "source": "oui_corrected",
            })

    return candidates


# Covers every OCR noise variant of the "S/N" label keyword.
# Ordered from most specific (slash variants) to most general (serial).
_SN_LABEL_RE = re.compile(
    r"s\s*/\s*n"                       # S/N  S/ N  S /N  S / N
    r"|s\s*\\\s*n"                     # S\N  (backslash variant)
    r"|s\s*i\s*n\b"                    # SIN  (slash misread as I by OCR)
    r"|s\s+n\b"                        # S N  (space between letters)
    r"|\bsn\b"                         # SN   (no separator)
    r"|\bserial\s*(?:number|no\.?)?",  # Serial Number | Serial No | Serial
    re.IGNORECASE,
)

# SN value: must start with a letter (distinguishes SN from IMEI/pure digits).
_SN_VALUE_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{4,}")


def collect_sn_candidates(
    block: str,
    ocr_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect serial number candidates from OCR text and barcode items.

    Handles same-line values (S/N: W51P...) and next-line values (label on one
    line, value on the next). All OCR noise variants of the keyword are matched.
    Pure-digit strings are rejected — those belong to IMEI, not SN.
    """
    candidates: List[Dict[str, Any]] = []
    ocr_seen: set = set()
    bc_seen: set = set()

    def _try_add(raw: str) -> None:
        val = fix_ocr_noise(raw.strip()).upper()
        if re.fullmatch(r"\d+", val):  # pure digits → IMEI, not SN
            return
        if len(val) >= 5 and val not in ocr_seen:
            ocr_seen.add(val)
            candidates.append({"value": val, "source": "ocr_text_rule"})

    lines = split_lines(block)

    for i, line in enumerate(lines):
        label_m = _SN_LABEL_RE.search(line)
        if not label_m:
            continue

        # Text on the same line after the keyword (strip leading colon/spaces)
        rest = line[label_m.end():].lstrip(": \t")

        if rest:
            val_m = _SN_VALUE_RE.search(rest)
            if val_m:
                _try_add(val_m.group(0))
        else:
            # Label-only line — check the very next line for the value
            if i + 1 < len(lines):
                val_m = _SN_VALUE_RE.search(lines[i + 1])
                if val_m:
                    _try_add(val_m.group(0))

    # Barcode candidates
    for item in ocr_items:
        if item.get("source") == "barcode" and item.get("barcode_type") == "sn":
            val = item["text"].upper()
            if val and val not in bc_seen:
                bc_seen.add(val)
                candidates.append({"value": val, "source": "barcode"})

    return candidates


def collect_imei_candidates(
    block: str,
    ocr_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    ocr_seen: set = set()
    hex_seen: set = set()
    bc_seen: set = set()

    for match in re.finditer(r"\bimei\s*:?\s*(\d{14,17})", block, re.IGNORECASE):
        val = match.group(1)
        if val not in ocr_seen:
            ocr_seen.add(val)
            candidates.append({"value": val, "source": "ocr_text_rule"})

    # C: allowlist hex pass yields digit-only runs — check for 14-17 digit sequences
    for item in ocr_items:
        if item.get("source") != "allowlist_hex":
            continue
        for d_match in re.finditer(r"\d{14,17}", item["text"]):
            val = d_match.group(0)
            if val not in hex_seen:
                hex_seen.add(val)
                candidates.append({"value": val, "source": "ocr_allowlist_hex"})

    for item in ocr_items:
        if item.get("source") == "barcode" and item.get("barcode_type") == "imei":
            val = item["text"]
            if val not in bc_seen:
                bc_seen.add(val)
                candidates.append({"value": val, "source": "barcode"})

    return candidates


def _resolve_candidate(
    candidates: List[Dict[str, Any]],
    avg_ocr_confidence: float,
    base_conf: float,
    field_label: str,
    warnings: List[str],
) -> Tuple[Optional[str], float, str]:
    """Pick best value from candidates.

    Priority (highest → lowest):
      barcode          — hardware-decoded, never wrong
      ocr_and_barcode  — OCR confirmed by barcode
      oui_corrected    — OCR corrected via known OUI lookup (principled fix)
      ocr_bbox_rule    — position-aware extraction
      ocr_text_rule    — flat-text regex

    OCR field confidence is always capped at avg_ocr_confidence to prevent
    falsely high scores when the global OCR pass was noisy.
    """
    bc_vals = [c for c in candidates if c.get("source") == "barcode"]
    oui_vals = [c for c in candidates if c.get("source") == "oui_corrected"]
    ocr_vals = [c for c in candidates if c.get("source") not in ("barcode", "oui_corrected")]

    # Best non-barcode OCR value: prefer bbox over text-rule
    def _ocr_priority(c: Dict[str, Any]) -> int:
        src = c.get("source", "")
        if "bbox" in src:
            return 1
        return 0

    ocr_vals_sorted = sorted(ocr_vals, key=_ocr_priority, reverse=True)
    ocr_val = ocr_vals_sorted[0]["value"] if ocr_vals_sorted else None
    ocr_src = ocr_vals_sorted[0].get("source", "ocr_text_rule") if ocr_vals_sorted else "ocr_text_rule"

    bc_val = bc_vals[0]["value"] if bc_vals else None
    oui_val = oui_vals[0]["value"] if oui_vals else None

    # Barcode cross-check (highest authority)
    if bc_val:
        if ocr_val and ocr_val == bc_val:
            return bc_val, min(base_conf, 0.99), "ocr_and_barcode"
        if oui_val and oui_val == bc_val:
            return bc_val, min(base_conf, 0.99), "ocr_and_barcode"
        if ocr_val and ocr_val != bc_val:
            warnings.append(
                f"OCR and barcode values differ for {field_label}. Technician must verify."
            )
        return bc_val, min(base_conf, 0.99), "barcode"

    # OUI-corrected: higher trust than raw OCR because the correction is
    # anchored to known manufacturer data, not a blind guess.
    if oui_val:
        original = oui_vals[0].get("original_ocr", "?")
        warnings.append(
            f"MAC corrected via OUI lookup ({original} -> {oui_val}). Please verify against label."
        )
        # Give slightly more confidence than raw OCR, but still flag for review
        corrected_conf = min(base_conf, max(avg_ocr_confidence + 0.10, 0.78))
        return oui_val, corrected_conf, "oui_corrected"

    # Raw OCR — cap confidence at global OCR quality
    if ocr_val:
        return ocr_val, min(avg_ocr_confidence, base_conf), ocr_src

    return None, 0.0, "not_found"


def normalize_equipment_ocr_text(text: str) -> str:
    """Fix OCR corruption patterns in equipment label text before field extraction."""
    # IMEI label: | and 1 are commonly misread as I
    text = re.sub(r"[|1]ME[I|1]\s*:", "IMEI:", text, flags=re.IGNORECASE)
    text = re.sub(r"IME[|1]\s*:", "IMEI:", text, flags=re.IGNORECASE)
    # S/N label: $ misread as S
    text = re.sub(r"\$\s*/\s*N\s*:", "S/N:", text)
    # MAC label: 4 misread as A
    text = re.sub(r"\bM4C\s*:", "MAC:", text, flags=re.IGNORECASE)
    return text


# -----------------------
# EQUIPMENT LABEL PARSING
# -----------------------

def parse_equipment_label(
    text: str,
    ocr_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    lines = split_lines(text)
    block = "\n".join(lines)
    block = normalize_equipment_ocr_text(block)
    warnings: List[str] = []

    # Global OCR quality — computed from regular (non-barcode) items.
    # Used to cap field confidence: a high regex match score means nothing
    # if the underlying OCR was unreliable.
    regular_items = [it for it in (ocr_items or []) if it.get("source") != "barcode"]
    avg_ocr_confidence = (
        sum(it["confidence"] for it in regular_items) / len(regular_items)
        if regular_items
        else 1.0
    )
    low_ocr = avg_ocr_confidence < 0.85

    if low_ocr:
        warnings.append(
            "Low OCR confidence. Technician must verify extracted values."
        )

    # Collect all candidates — collectors now accept full ocr_items and
    # split internally by source (barcode / allowlist_hex / ocr_text_rule).
    all_items = ocr_items or []
    mac_candidates = collect_mac_candidates(block, all_items)
    sn_candidates = collect_sn_candidates(block, all_items)
    imei_candidates = collect_imei_candidates(block, all_items)

    # B: bbox-guided extraction — finds the value item physically nearest to
    # the label item ("MAC:", "S/N:") using bounding box positions.
    # More reliable than flat-text regex when OCR reads items out of order.
    regular_items = [it for it in all_items if it.get("source") not in ("barcode", "allowlist_hex")]

    bbox_mac = extract_by_bbox_label(
        regular_items,
        [r"\bmac\b"],
        lambda t: normalize_mac(fix_hex_noise(t)),
    )
    if bbox_mac:
        existing_ocr_vals = {c["value"] for c in mac_candidates if c.get("source") not in ("barcode",)}
        if bbox_mac["value"] not in existing_ocr_vals:
            mac_candidates.insert(0, {
                "value": bbox_mac["value"],
                "raw": bbox_mac["text"],
                "source": "ocr_bbox_rule",
            })

    bbox_sn = extract_by_bbox_label(
        regular_items,
        # Same noise variants as _SN_LABEL_RE so bbox search matches SIN, S N, etc.
        [r"s\s*/\s*n", r"s\s*i\s*n\b", r"s\s+n\b", r"\bsn\b", r"\bserial\b"],
        lambda t: (
            fix_ocr_noise(t.strip()).upper()
            if (
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-]{4,}", t.strip())
                and not re.fullmatch(r"\d+", t.strip())  # reject pure-digit (IMEI)
            )
            else None
        ),
    )
    if bbox_sn:
        existing_sn_vals = {c["value"] for c in sn_candidates if c.get("source") != "barcode"}
        if bbox_sn["value"] not in existing_sn_vals:
            sn_candidates.insert(0, {
                "value": bbox_sn["value"],
                "source": "ocr_bbox_rule",
            })

    # Model
    model_match = extract_first_match(
        [r"^model\s*:?\s*(.+)"],
        block,
        re.IGNORECASE | re.MULTILINE,
    )
    model_value = clean_model(model_match.group(1)) if model_match else None

    # Prioritize SN candidates whose value starts with the model suffix
    # (e.g. model="ZLT W51" → prefix="W51"; "W51P740912346" rises to top).
    if model_value and sn_candidates:
        parts = model_value.split()
        if len(parts) >= 2:
            prefix = parts[-1].upper()
            sn_candidates.sort(
                key=lambda c: (
                    0 if c["value"].upper().startswith(prefix) else 1,
                    0 if c.get("source") == "barcode" else 1,
                )
            )

    # Warn when an SN keyword was detected but extraction still failed
    if not sn_candidates and _SN_LABEL_RE.search(block):
        warnings.append(
            "Serial keyword detected but value could not be extracted"
        )

    # Resolve best value per field, cross-checking OCR vs barcode
    mac_value, mac_conf, mac_source = _resolve_candidate(
        mac_candidates, avg_ocr_confidence, 0.95, "MAC address", warnings
    )
    sn_value, sn_conf, sn_source = _resolve_candidate(
        sn_candidates, avg_ocr_confidence, 0.95, "serial number", warnings
    )
    imei_value, imei_conf, imei_source = _resolve_candidate(
        imei_candidates, avg_ocr_confidence, 0.90, "IMEI", warnings
    )

    # OUI boost: if the final MAC (from any source) matches a known ZTE/TOZED
    # OUI, add a small confidence boost — it confirms the address is real.
    if mac_value and mac_value[:6].upper() in KNOWN_ROUTER_OUIS:
        if mac_source not in ("oui_corrected", "barcode", "ocr_and_barcode"):
            mac_conf = min(mac_conf + 0.05, 0.99)
            mac_source = mac_source + "_oui_validated"

    # Missing field warnings
    if mac_value is None:
        warnings.append("MAC address not found")
    if sn_value is None:
        if _SN_LABEL_RE.search(block):
            warnings.append("Serial keyword detected but value could not be extracted")
        else:
            warnings.append("Serial number not found")
    if imei_value is None:
        warnings.append("IMEI not found")

    # Additional noisy-OCR warning when key fields were found but quality is low
    if (mac_value or sn_value) and low_ocr:
        warnings.append(
            "Please verify MAC and serial number against the photo."
        )

    needs_review = (
        mac_value is None
        or sn_value is None
        or low_ocr
        or len(mac_candidates) > 1
        or len(sn_candidates) > 1
        or any("differ" in w for w in warnings)
    )

    return {
        "type": "EQUIPMENT_LABEL",
        "model": field_result(model_value, 0.90 if model_value else 0.0, "text_rule"),
        "macAddress": field_result(mac_value, mac_conf, mac_source),
        "serialNumber": field_result(sn_value, sn_conf, sn_source),
        "imei": field_result(imei_value, imei_conf, imei_source),
        "warnings": warnings,
        "needsReview": needs_review,
        "requiresConfirmation": True,
        "candidates": {
            "macAddress": mac_candidates,
            "serialNumber": sn_candidates,
            "imei": imei_candidates,
        },
    }


# -----------------------
# REVIEW LOGIC
# -----------------------

def router_needs_review(data: Dict[str, Any]) -> bool:
    """Router screen: review when a critical field is missing or out of range."""
    for field in CRITICAL_ROUTER_FIELDS:
        entry = data.get(field)
        if entry is None or entry.get("value") is None:
            return True
        if entry.get("rangeValid") is False:
            return True
    return False


def _router_field_valid(entry: Optional[Dict[str, Any]]) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("value") is not None
        and entry.get("rangeValid") is not False
    )


def router_critical_fields_valid(parsed: Dict[str, Any]) -> bool:
    """True when every critical router field extracted with a plausible value.

    Used by the OCR engine to decide whether the EN-only pass is sufficient
    or the AR+EN fallback pass must run.
    """
    if parsed.get("type") != "router_signal_screen":
        return False
    return all(
        _router_field_valid(parsed.get(field)) for field in CRITICAL_ROUTER_FIELDS
    )


def merge_router_results(
    en_result: Dict[str, Any],
    ar_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge the EN-pass and AR+EN-pass router parses field by field.

    The EN recognition model is authoritative on Latin text: a valid EN value
    is never overwritten. The AR+EN pass only fills fields the EN pass missed
    or that failed range validation. Each field's source records which pass
    produced it ("@en_pass" / "@ar_en_pass").
    """
    merged: Dict[str, Any] = {"type": "router_signal_screen"}
    for field in ROUTER_RESULT_FIELDS:
        en_entry = en_result.get(field) if en_result else None
        ar_entry = ar_result.get(field) if ar_result else None

        if _router_field_valid(en_entry):
            entry, ocr_pass = en_entry, "en"
        elif _router_field_valid(ar_entry):
            entry, ocr_pass = ar_entry, "ar_en"
        elif en_entry is not None and en_entry.get("value") is not None:
            # Out-of-range in both passes — keep EN, confidence already <= 0.4
            entry, ocr_pass = en_entry, "en"
        elif ar_entry is not None and ar_entry.get("value") is not None:
            entry, ocr_pass = ar_entry, "ar_en"
        else:
            entry = en_entry or ar_entry or field_result(None, 0.0, "not_found")
            ocr_pass = "en"

        entry = dict(entry)
        entry["source"] = f"{entry.get('source', 'unknown')}@{ocr_pass}_pass"
        merged[field] = entry

    merged["needsReview"] = router_needs_review(merged)
    return merged


def needs_review(data: Dict[str, Any]) -> bool:
    for key, value in data.items():
        if key in ["type", "needsReview"]:
            continue

        if isinstance(value, dict):
            if value.get("value") is None:
                return True

            if value.get("confidence", 0) < 0.65:
                return True

    return False


def flatten_values(data: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}

    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
        else:
            flat[key] = value

    return flat


# -----------------------
# MAIN
# -----------------------

def extract_structured_data(
    text: str,
    ocr_items: Optional[List[Dict[str, Any]]] = None,
    include_confidence: bool = True,
) -> Dict[str, Any]:
    doc_type = detect_document_type(text)

    if doc_type == "router_signal_screen":
        result = parse_router_screen(text, ocr_items=ocr_items)

    elif doc_type == "speedtest_screen":
        result = parse_speedtest_screen(text, ocr_items)

    elif doc_type == "EQUIPMENT_LABEL":
        result = parse_equipment_label(text, ocr_items=ocr_items)

    else:
        result = {
            "type": "unknown",
            "needsReview": True,
            "message": "Could not detect document type",
        }

    if include_confidence:
        return result

    return flatten_values(result)


def extract_from_text(text: str) -> Dict[str, Any]:
    return extract_structured_data(text)