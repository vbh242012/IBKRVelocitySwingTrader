#!/usr/bin/env python
"""Run the application premarket universe prefilter once.

This connects to IBKR through the normal VelocityEngine, runs only the
historical universe sieve, prints a compact summary, and disconnects. It does
not start the trading loop or place orders.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing env file: {path}")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Velocity premarket universe prefilter once.")
    parser.add_argument("--profile", choices=("paper", "live"), default="live")
    parser.add_argument("--client-id", type=int, default=91, help="Separate IB API client id for this one-shot run.")
    parser.add_argument("--force", action="store_true", help="Delete today's cache first and rescan.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Debug cap; 0 means full configured universe.")
    parser.add_argument("--sleep-sec", type=float, default=None, help="Override historical-request pacing sleep.")
    parser.add_argument("--progress-every", type=int, default=None, help="Override progress checkpoint interval.")
    parser.add_argument(
        "--ignore-entry-cutoff",
        action="store_true",
        help="Keep scanning even after the live entry window has opened.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_file = PROJECT_DIR / f".env.{args.profile}.local"
    _load_env_file(env_file)

    os.environ["VELOCITY_PROFILE"] = args.profile
    os.environ.setdefault("VELOCITY_BASE_DIR", str(PROJECT_DIR / "runtime" / args.profile))
    os.environ["VELOCITY_IB_CLIENT_ID"] = str(args.client_id)
    os.environ["VELOCITY_APP_PREFILTER_ENABLED"] = "1"
    if args.max_symbols is not None:
        os.environ["VELOCITY_APP_SCANNER_MAX_SYMBOLS"] = str(args.max_symbols)
    if args.sleep_sec is not None:
        os.environ["VELOCITY_APP_PREFILTER_HISTORY_SLEEP_SEC"] = str(args.sleep_sec)
    if args.progress_every is not None:
        os.environ["VELOCITY_APP_PREFILTER_PROGRESS_EVERY"] = str(args.progress_every)
    if args.ignore_entry_cutoff:
        os.environ["VELOCITY_APP_PREFILTER_STOP_AT_ENTRY_START"] = "0"

    sys.path.insert(0, str(PROJECT_DIR))
    from src.config import APP_PREFILTER_CACHE_FILE, APP_SCANNER_MAX_SYMBOLS, STRATEGY_PROFILE
    from src.engine import VelocityEngine

    cache_path = Path(APP_PREFILTER_CACHE_FILE)
    if args.force and cache_path.exists():
        cache_path.unlink()

    print(
        "Prefilter run starting: "
        f"profile={args.profile} strategy={STRATEGY_PROFILE} "
        f"client_id={args.client_id} max_symbols={APP_SCANNER_MAX_SYMBOLS or 'full'} "
        f"cache={cache_path}"
    )
    engine = VelocityEngine()
    try:
        payload = engine._run_premarket_universe_prefilter()
        stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
        print("Prefilter run complete.")
        print(f"  status     : {payload.get('status') if isinstance(payload, dict) else 'unknown'}")
        print(f"  processed  : {stats.get('processed', 0)} / {stats.get('universe', 0)}")
        print(f"  candidates : {stats.get('candidates', 0)}")
        print(f"  rejected   : {stats.get('rejected', 0)}")
        print(f"  cache      : {cache_path}")
        return 0
    finally:
        try:
            engine.ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
