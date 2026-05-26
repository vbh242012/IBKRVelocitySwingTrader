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

HOST="${VELOCITY_IB_HOST:-127.0.0.1}"
PORT="${VELOCITY_IB_PORT:-4002}"
LOG_FILE="${VELOCITY_BASE_DIR}/logs/trading_engine.log"
LOCK_FILE="${VELOCITY_BASE_DIR}/velocity_engine.lock"

echo "Project: ${PROJECT_DIR}"
echo "Profile: ${PROFILE}"
echo "Mode: ${VELOCITY_TRADING_MODE:-paper}"
echo "Runtime: ${VELOCITY_BASE_DIR}"
echo "IB API: ${HOST}:${PORT}"

if [[ -f "${LOCK_FILE}" ]]; then
  PID="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    echo "Engine: running (pid ${PID})"
  else
    echo "Engine: lock exists but pid is not running (${PID:-empty})"
  fi
else
  echo "Engine: no lock file found"
fi

if timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
  echo "IB API port: reachable"
else
  echo "IB API port: not reachable"
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo
  echo "Recent engine log:"
  tail -n 40 "${LOG_FILE}"
else
  echo "No trading engine log found at ${LOG_FILE}"
fi
