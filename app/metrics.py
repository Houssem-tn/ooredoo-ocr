"""Prometheus metrics for the OCR service.

A focused set, not a metric per code path — see OBSERVABILITY_BACKUP.md
"OCR metrics" for why exactly these. No label ever carries a filename,
document ID, or case ID (see that doc's "Sensitive-data protections" —
high-cardinality/identifying labels defeat Prometheus's storage model as
much as they leak information). Exposed at GET /metrics — internal-only:
the OCR Kubernetes Service is ClusterIP and is never referenced by the
Ingress (see deploy/helm/ooredoo/templates/ingress.yaml), so this is
never reachable from outside the cluster.
"""
from prometheus_client import Counter, Gauge, Histogram

ocr_requests_total = Counter(
    "ocr_requests_total",
    "OCR extraction requests, by outcome.",
    ["endpoint", "status"],  # status: success | failure | invalid_input
)

ocr_request_duration_seconds = Histogram(
    "ocr_request_duration_seconds",
    "Time spent in the OCR extraction pipeline (image decode through structured-data extraction), per request.",
    ["endpoint"],
    # EasyOCR on CPU: tens of ms for trivial input up to the ~15s typical
    # case noted in ocr_engine.py's own sizing comment, occasionally more
    # for the two-pass (EN + AR+EN fallback) router-screen path.
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 60),
)

ocr_active_requests = Gauge(
    "ocr_active_requests",
    "OCR extraction requests currently being processed.",
)

ocr_invalid_input_total = Counter(
    "ocr_invalid_input_total",
    "Rejected uploads (bad extension, empty file, unreadable/corrupted image), by reason.",
    ["reason"],
)

ocr_model_loaded = Gauge(
    "ocr_model_loaded",
    "1 once both EasyOCR Reader models have finished loading, 0 before/if that fails.",
)

ocr_temp_file_cleanup_failures_total = Counter(
    "ocr_temp_file_cleanup_failures_total",
    "Times a temp upload file under temp_uploads/ could not be deleted after processing (see app/utils.py delete_file_safely).",
)
