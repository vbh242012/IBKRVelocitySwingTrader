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
EXAMPLE_FILE="${PROJECT_DIR}/.env.${PROFILE}.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it from ${EXAMPLE_FILE} and keep it out of git." >&2
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

if [[ "${PROFILE}" == "live" ]]; then
  if [[ "${VELOCITY_TRADING_MODE:-}" != "live" ]]; then
    echo "Refusing live start: VELOCITY_TRADING_MODE must be live in ${ENV_FILE}." >&2
    exit 3
  fi
  if [[ "${VELOCITY_LIVE_TRADING_ACK:-}" != "I_UNDERSTAND_LIVE_RISK" ]]; then
    echo "Refusing live start: set VELOCITY_LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_RISK in ${ENV_FILE}." >&2
    exit 3
  fi
fi

export VELOCITY_PROFILE="${PROFILE}"
export VELOCITY_BASE_DIR="${VELOCITY_BASE_DIR:-${PROJECT_DIR}/runtime/${PROFILE}}"

cd "${PROJECT_DIR}"
mkdir -p logs "${VELOCITY_BASE_DIR}/logs"

exec .venv/bin/python auto_trader.py
