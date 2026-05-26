#!/usr/bin/env bash
set -euo pipefail

# Starts the read-only dashboard for a profile.
#
# Usage:
#   ./scripts/start_dashboard.sh paper
#   ./scripts/start_dashboard.sh live
#
# The dashboard reads the selected profile's runtime folder. This keeps paper
# dashboard state separate from live dashboard state.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-paper}"

# Only supported profiles may be used because the profile name determines which
# local env file and runtime directory are loaded.
case "${PROFILE}" in
  paper|live) ;;
  *)
    echo "Usage: $0 [paper|live]" >&2
    exit 2
    ;;
esac

ENV_FILE="${PROJECT_DIR}/.env.${PROFILE}.local"

# Dashboard can start even when an env file is missing, but it will then use
# safe localhost defaults and the default runtime folder for the profile.
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

export VELOCITY_PROFILE="${PROFILE}"
export VELOCITY_BASE_DIR="${VELOCITY_BASE_DIR:-${PROJECT_DIR}/runtime/${PROFILE}}"

cd "${PROJECT_DIR}"
mkdir -p logs "${VELOCITY_BASE_DIR}/logs"

# Host/port are configurable per profile so paper and live dashboards can run
# side by side without fighting for the same port.
exec .venv/bin/python dashboard_server.py \
  --host "${VELOCITY_DASHBOARD_HOST:-127.0.0.1}" \
  --port "${VELOCITY_DASHBOARD_PORT:-8080}"
