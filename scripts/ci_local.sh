#!/usr/bin/env bash
# Reproduce GitHub Actions CI for kazenai-core hermetically (no host .env leakage).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Clear host policy env that would diverge from Actions
unset KAZENAI_UNKNOWN_MODEL_POLICY KAZENAI_DEPLOYMENT_MODE KAZENAI_ENV KAZENAI_PRICING_JSON || true

VENV="${CI_VENV:-/tmp/kazenai-core-ci-local}"
PYTHON="${PYTHON:-python3.11}"
rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install ./vendor/kazen_event_schema-0.6.0-py3-none-any.whl
pip install ./vendor/kazenai_contracts-0.1.0-py3-none-any.whl
pip install -e ".[dev]"
pytest tests/ -q --cov=kazenai --cov-report=term-missing --cov-fail-under=80
python scripts/export_kazenevent_schema.py
echo "PASS local CI mirror"
