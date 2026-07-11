#!/usr/bin/env bash
# Reproduces .github/workflows/ci.yml's `validate` job locally — same
# ruff/pytest commands CI runs, no separate logic to keep in sync.
# Run from an activated venv with requirements.txt + ruff + pytest + httpx
# installed (see README.md), or point PYTHON at one:
#
# Usage: ./scripts/ci-local.sh   (PYTHON=venv/Scripts/python ./scripts/ci-local.sh on Windows)
#
# httpx is required for the import/startup smoke test below
# (fastapi.testclient.TestClient needs it) even though it's not a runtime
# dependency — this bit CI on its first real run (missing from the
# workflow's own pip install list, since local testing had it installed
# incidentally via an unrelated `pip install fastapi[all]`).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

echo "==> Lint (ruff)"
"$PYTHON" -m ruff check app/ tests/ --config ruff.toml

echo "==> Tests"
"$PYTHON" -m pytest tests/ -q

echo "==> Import / startup smoke test (mocked EasyOCR reader — no model download)"
"$PYTHON" -c "
from unittest.mock import patch
from fastapi.testclient import TestClient

with patch('easyocr.Reader'):
    from app.main import app

client = TestClient(app)
resp = client.get('/')
assert resp.status_code == 200, resp.text
print('OK:', resp.json())
"

if command -v docker >/dev/null 2>&1; then
  echo "==> Docker build (validation only, no push — bakes real EasyOCR models in, slow on first run)"
  docker build -t ooredoo-ocr:ci-local .
else
  echo "==> Docker not installed locally — skipping image build (CI still runs it)"
fi

echo "==> Done."
