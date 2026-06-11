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
# shell supervisor covers two different failure classes:
#   1. auto_trader.py exits because of an unhandled runtime/environment failure.
#   2. auto_trader.py stays alive but stops writing fresh dashboard heartbeat
#      data, which makes the dashboard look frozen while the process is stuck.
# Keep it enabled for unattended paper/live runs; disable only when debugging in
# the foreground.
AUTO_RESTART="${VELOCITY_TRADER_AUTO_RESTART:-1}"
RESTART_DELAY_SEC="${VELOCITY_TRADER_RESTART_DELAY_SEC:-30}"
DISABLE_RESTART_FILE="${VELOCITY_BASE_DIR}/DISABLE_AUTO_RESTART"

# Heartbeat watchdog.  The engine writes dashboard_data.json every cycle.  If
# that file stops moving for too long after startup grace, the supervisor kills
# the child and lets the normal restart path bring it back.  Defaults are
# deliberately conservative so a slow IBKR scanner cycle is not mistaken for a
# hang.
WATCHDOG_ENABLED="${VELOCITY_TRADER_WATCHDOG_ENABLED:-1}"
WATCHDOG_HEARTBEAT_FILE="${VELOCITY_TRADER_HEARTBEAT_FILE:-${VELOCITY_BASE_DIR}/dashboard_data.json}"
WATCHDOG_STALE_SEC="${VELOCITY_TRADER_STALE_SEC:-600}"
WATCHDOG_INTERVAL_SEC="${VELOCITY_TRADER_WATCHDOG_INTERVAL_SEC:-15}"
WATCHDOG_STARTUP_GRACE_SEC="${VELOCITY_TRADER_WATCHDOG_STARTUP_GRACE_SEC:-900}"

case "${AUTO_RESTART}" in
  1|true|TRUE|yes|YES|on|ON) AUTO_RESTART=1 ;;
  *) AUTO_RESTART=0 ;;
esac

case "${WATCHDOG_ENABLED}" in
  1|true|TRUE|yes|YES|on|ON) WATCHDOG_ENABLED=1 ;;
  *) WATCHDOG_ENABLED=0 ;;
esac

positive_int_or_default() {
  # Accept only positive integers for timing knobs.  Bad local env values fall
  # back to safe defaults instead of breaking the supervisor.
  local value="$1"
  local default="$2"
  if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${default}"
  fi
}

RESTART_DELAY_SEC="$(positive_int_or_default "${RESTART_DELAY_SEC}" 30)"
WATCHDOG_STALE_SEC="$(positive_int_or_default "${WATCHDOG_STALE_SEC}" 600)"
WATCHDOG_INTERVAL_SEC="$(positive_int_or_default "${WATCHDOG_INTERVAL_SEC}" 15)"
WATCHDOG_STARTUP_GRACE_SEC="$(positive_int_or_default "${WATCHDOG_STARTUP_GRACE_SEC}" 900)"

et_now() {
  # Keep supervisor stderr timestamps aligned with the trading engine logs.
  TZ=America/New_York date '+%Y-%m-%dT%H:%M:%S%z %Z'
}

child_is_running() {
  # `kill -0` is true for zombies, so include process state before deciding that
  # the child is genuinely still alive.
  [[ -n "${child_pid}" ]] || return 1
  kill -0 "${child_pid}" 2>/dev/null || return 1
  local child_stat
  child_stat="$(ps -o stat= -p "${child_pid}" 2>/dev/null || true)"
  [[ -n "${child_stat}" && "${child_stat}" != Z* ]]
}

file_mtime_epoch() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  stat -c '%Y' "${path}" 2>/dev/null
}

stop_stale_child_if_needed() {
  local child_started_epoch="$1"
  [[ "${WATCHDOG_ENABLED}" == "1" ]] || return 1
  child_is_running || return 1

  local now_epoch runtime_sec heartbeat_mtime age_sec
  now_epoch="$(date +%s)"
  runtime_sec=$((now_epoch - child_started_epoch))
  if (( runtime_sec < WATCHDOG_STARTUP_GRACE_SEC )); then
    return 1
  fi

  heartbeat_mtime="$(file_mtime_epoch "${WATCHDOG_HEARTBEAT_FILE}" || true)"
  if [[ -z "${heartbeat_mtime}" ]]; then
    age_sec="${runtime_sec}"
  else
    age_sec=$((now_epoch - heartbeat_mtime))
  fi

  if (( age_sec <= WATCHDOG_STALE_SEC )); then
    return 1
  fi

  echo "$(et_now) watchdog: ${WATCHDOG_HEARTBEAT_FILE} stale for ${age_sec}s; terminating trader pid ${child_pid}." >&2
  kill "${child_pid}" 2>/dev/null || return 0

  # Give Python a short chance to release IB subscriptions and the instance
  # lock.  Escalate only if it refuses to exit.
  for _ in {1..20}; do
    child_is_running || return 0
    sleep 1
  done

  echo "$(et_now) watchdog: trader pid ${child_pid} ignored SIGTERM; sending SIGKILL." >&2
  kill -KILL "${child_pid}" 2>/dev/null || true
  return 0
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
  child_started_epoch="$(date +%s)"

  while child_is_running; do
    if [[ -f "${DISABLE_RESTART_FILE}" ]]; then
      echo "$(et_now) auto-restart disabled by ${DISABLE_RESTART_FILE}; stopping trader." >&2
      stop_child
    fi
    stop_stale_child_if_needed "${child_started_epoch}" || true
    sleep "${WATCHDOG_INTERVAL_SEC}"
  done

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
