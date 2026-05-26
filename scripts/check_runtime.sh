#!/usr/bin/env bash
set -euo pipefail

# Lightweight operational health check for one profile.
#
# Usage:
#   ./scripts/check_runtime.sh paper
#   ./scripts/check_runtime.sh live
#
# This does not connect through the IB API. It checks:
#   - whether the engine lock PID is still alive
#   - whether the configured IB Gateway API port is reachable
#   - the latest trading-engine log lines for the selected runtime folder
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-paper}"

# Keep profile names explicit because they map directly to env files and runtime
# paths.
case "${PROFILE}" in
  paper|live) ;;
  *)
    echo "Usage: $0 [paper|live]" >&2
    exit 2
    ;;
esac

ENV_FILE="${PROJECT_DIR}/.env.${PROFILE}.local"

# Source local profile values if present. Missing env files are tolerated here
# so the checker can still report default paths/ports.
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

# The lock file is created by auto_trader.py. A stale lock is reported clearly
# because it usually means the engine was killed or crashed outside its cleanup.
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

# A reachable TCP port means Gateway is listening. It does not prove that market
# data subscriptions are healthy, but it is the fastest liveness check.
if timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
  echo "IB API port: reachable"
else
  echo "IB API port: not reachable"
fi

# Keep the output compact enough to inspect over SSH or from a terminal pane.
if [[ -f "${LOG_FILE}" ]]; then
  echo
  echo "Recent engine log:"
  tail -n 40 "${LOG_FILE}"
else
  echo "No trading engine log found at ${LOG_FILE}"
fi
