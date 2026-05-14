#!/usr/bin/env bash
# Run the FastAPI app from the project root (Linux/macOS).
# Usage: ./scripts/run_api.sh
#        ./scripts/run_api.sh --reload
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="."
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8080 "$@"
else
  exec python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8080 "$@"
fi
