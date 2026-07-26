"""Structured logging for the OCR service.

JSON output (LOG_JSON=true, the default under Kubernetes — see
Pfe/deploy/helm/ooredoo/values.yaml's logging.json) or plain text
(local dev), controlled the same way the backend's logback-spring.xml
switches format — see that file's own comment for why this exists at
all: centralized logging (Loki) needs one parseable shape per service,
not five different ad-hoc `print()` styles.

Standard fields match OBSERVABILITY_BACKUP.md's "Logging fields" table
exactly: timestamp, level, service, environment, namespace, pod,
requestId, message. Never includes recognized document text or uploaded
file contents — every log call site in this service was audited for
that when this was introduced (see app/main.py, app/utils.py,
app/ocr_engine.py: only filenames, sizes, shapes, confidence scores, and
pass/fail outcomes are logged, never OCR output).
"""
import json
import logging
import os
import sys

from app.request_context import get_request_id

SERVICE_NAME = "ooredoo-ocr"

# Every attribute a plain logging.LogRecord carries by default — anything
# else present on a record came from a caller's `extra={...}` and should
# be surfaced as its own JSON field (e.g. logger.info("...", extra={
# "filename": ..., "confidence": ...}) in app/main.py) rather than
# silently dropped, which is what a bare logging.Formatter would do.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "request_id"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": SERVICE_NAME,
            "environment": os.environ.get("APP_ENV", "dev"),
            "namespace": os.environ.get("POD_NAMESPACE", ""),
            "pod": os.environ.get("POD_NAME", ""),
            "logger": record.name,
            "requestId": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class RequestIdLogFilter(logging.Filter):
    """Injects the current request's correlation ID (see
    app/request_context.py) into every LogRecord, so both JSON and plain
    formatters can include it without every call site passing it manually.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    json_enabled = os.environ.get("LOG_JSON", "false").strip().lower() == "true"
    level_name = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdLogFilter())
    if json_enabled:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s - %(message)s")
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn's own loggers otherwise bypass this handler entirely.
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uvicorn_logger).handlers = [handler]
        logging.getLogger(uvicorn_logger).propagate = False


logger = logging.getLogger("ocr")
