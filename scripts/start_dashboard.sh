#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-paper}"

case "${PROFILE}" in
  paper|live) ;;
  *)
    echo "Usage: $0 [paper|live]" >&2
    exit 2
    ;;
esac

ENV_FILE="${PROJECT_DIR}/.env.${PROFILE}.local"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

export VELOCITY_PROFILE="${PROFILE}"
export VELOCITY_BASE_DIR="${VELOCITY_BASE_DIR:-${PROJECT_DIR}/runtime/${PROFILE}}"

cd "${PROJECT_DIR}"
mkdir -p logs "${VELOCITY_BASE_DIR}/logs"

exec .venv/bin/python dashboard_server.py \
  --host "${VELOCITY_DASHBOARD_HOST:-127.0.0.1}" \
  --port "${VELOCITY_DASHBOARD_PORT:-8080}"
