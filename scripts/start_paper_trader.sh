#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env.paper.local"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it from .env.paper.example and keep it out of git." >&2
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

cd "${PROJECT_DIR}"
mkdir -p logs

exec .venv/bin/python auto_trader.py
