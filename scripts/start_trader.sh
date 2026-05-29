#!/usr/bin/env bash
set -euo pipefail

# Starts the trading engine for exactly one runtime profile.
#
# Usage:
#   ./scripts/start_trader.sh paper
#   ./scripts/start_trader.sh live
#
# The profile chooses:
#   - which local env file is loaded: .env.paper.local or .env.live.local
#   - which IB Gateway port is used
#   - which runtime folder stores state, logs, dashboard data, and lock files
#
# Live mode has an extra acknowledgement gate here and inside Python. This is
# intentional: switching from paper to live must be deliberate, never accidental.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-paper}"

# Keep the accepted profile names small and explicit. These names are also used
# in runtime paths, so free-form profile names are not allowed.
case "${PROFILE}" in
  paper|live) ;;
  *)
    echo "Usage: $0 [paper|live]" >&2
    exit 2
    ;;
esac

ENV_FILE="${PROJECT_DIR}/.env.${PROFILE}.local"
EXAMPLE_FILE="${PROJECT_DIR}/.env.${PROFILE}.example"

# Local env files may contain account-specific ports, dashboard ports, webhook
# URLs, and IBC launcher paths. They are intentionally ignored by git.
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it from ${EXAMPLE_FILE} and keep it out of git." >&2
  exit 2
fi

# Export every variable from the profile env file so src/config.py can read it
# with os.getenv() without needing a Python dotenv dependency.
set -a
source "${ENV_FILE}"
set +a

# Live profile protection. The Python engine also validates this, but checking
# before startup gives a fast, obvious failure and avoids even launching Gateway.
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

# VELOCITY_BASE_DIR isolates paper and live state. This prevents a paper
# position/state file from ever being reused by live trading.
export VELOCITY_PROFILE="${PROFILE}"
export VELOCITY_BASE_DIR="${VELOCITY_BASE_DIR:-${PROJECT_DIR}/runtime/${PROFILE}}"

cd "${PROJECT_DIR}"
mkdir -p logs "${VELOCITY_BASE_DIR}/logs"

# Optional process supervision.  The Python engine already handles IB API
# reconnects and can auto-start IBC/Gateway when the API port is down.  This
# shell loop covers the different failure class: the Python process itself exits
# because of an unhandled runtime/environment failure.  Keep it enabled for
# unattended paper/live runs; disable only when debugging in the foreground.
AUTO_RESTART="${VELOCITY_TRADER_AUTO_RESTART:-1}"
RESTART_DELAY_SEC="${VELOCITY_TRADER_RESTART_DELAY_SEC:-30}"
DISABLE_RESTART_FILE="${VELOCITY_BASE_DIR}/DISABLE_AUTO_RESTART"

case "${AUTO_RESTART}" in
  1|true|TRUE|yes|YES|on|ON) AUTO_RESTART=1 ;;
  *) AUTO_RESTART=0 ;;
esac

et_now() {
  # Keep supervisor stderr timestamps aligned with the trading engine logs.
  TZ=America/New_York date '+%Y-%m-%dT%H:%M:%S%z %Z'
}

if [[ "${AUTO_RESTART}" == "0" ]]; then
  # Debug mode: replace the shell with Python so the terminal/process manager
  # observes the real trading process directly.
  exec .venv/bin/python auto_trader.py
fi

child_pid=""

stop_child() {
  # Graceful shutdown path.  If the supervisor receives SIGTERM/SIGINT, forward
  # it to the engine and wait so auto_trader.py can release its instance lock.
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit 0
}

trap stop_child INT TERM

while true; do
  if [[ -f "${DISABLE_RESTART_FILE}" ]]; then
    echo "$(et_now) auto-restart disabled by ${DISABLE_RESTART_FILE}; exiting." >&2
    exit 0
  fi

  .venv/bin/python auto_trader.py &
  child_pid="$!"
  # `wait` returns the child's exit status.  With `set -e`, a non-zero child
  # status would otherwise terminate this supervisor before it can restart.
  set +e
  wait "${child_pid}"
  status="$?"
  set -e
  child_pid=""

  if [[ -f "${DISABLE_RESTART_FILE}" ]]; then
    echo "$(et_now) trader exited with status ${status}; restart disabled." >&2
    exit "${status}"
  fi

  echo "$(et_now) trader exited with status ${status}; restarting in ${RESTART_DELAY_SEC}s." >&2
  sleep "${RESTART_DELAY_SEC}"
done
