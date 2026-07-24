"""Correlation-ID propagation for the OCR service.

Accepts the X-Request-ID the backend forwards (see
APPSTOCK/PFE-Ooredoo's RequestCorrelationFilter, which either passes
through nginx's own $request_id or a client-supplied header — see
Pfe-Oreedoo-Front/nginx.conf), or generates one if the service is called
directly (e.g. during local/manual testing). Never derived from or
containing any user identifier — see OBSERVABILITY_BACKUP.md
"Correlation IDs".
"""
import uuid
from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(value: str) -> None:
    _request_id_var.set(value if value else uuid.uuid4().hex)


def new_request_id() -> str:
    return uuid.uuid4().hex
