"""
Pytest configuration:
  1. Suppress engine logger so CRITICAL/WARNING messages from mocked failure
     paths never bleed into the live logs/ directory.
  2. Redirect all production file paths (state, dashboard, equity history, log)
     to per-test tmp directories so tests can never corrupt live data files.
"""

import logging
import pytest
import src.config as cfg
import src.engine as eng
import src.engine_base as eng_base
import src.engine_orders as eng_orders


@pytest.fixture(autouse=True)
def silence_engine_logger():
    """Suppress the engine logger completely for every test.

    - Raises the level to CRITICAL+1 so nothing is emitted.
    - Temporarily removes file handlers so the live trading_engine.log is not
      polluted with test-fixture output (e.g. AAPL $150, mock equity values).
      Handlers are restored after the test so module-level state is unchanged.
    """
    logger = logging.getLogger("VelocityEngine")
    original_level = logger.level
    original_handlers = logger.handlers[:]
    file_handlers = [h for h in logger.handlers if hasattr(h, 'baseFilename')]
    for h in file_handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.CRITICAL + 1)
    yield
    logger.setLevel(original_level)
    for h in file_handlers:
        logger.addHandler(h)


@pytest.fixture(autouse=True)
def isolate_production_files(tmp_path):
    """
    Redirect STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, READINESS_FILE,
    HEALTH_REPORT_FILE, LOG_DIR and LOG_FILE to a per-test temp directory so
    tests can never touch live data or log files.

    Also patches the same names in src.engine_base (where the I/O methods
    actually read these constants) and src.engine_orders (for stop-confirm
    timeouts), so the code paths in the mixin modules use the tmp paths.
    """
    orig_state   = cfg.STATE_FILE
    orig_dash    = cfg.DASHBOARD_FILE
    orig_equity  = cfg.EQUITY_HIST_FILE
    orig_ready   = cfg.READINESS_FILE
    orig_health  = cfg.HEALTH_REPORT_FILE
    orig_halt    = cfg.HALT_FILE
    orig_force   = cfg.FORCE_EXIT_FILE
    orig_log_dir = cfg.LOG_DIR
    orig_log     = cfg.LOG_FILE
    orig_stop_confirm_timeout = eng.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC
    orig_stop_confirm_poll    = eng.PROTECTIVE_STOP_CONFIRM_POLL_SEC

    # Save engine_base originals (the module where I/O methods actually live)
    orig_base_state   = eng_base.STATE_FILE
    orig_base_dash    = eng_base.DASHBOARD_FILE
    orig_base_equity  = eng_base.EQUITY_HIST_FILE
    orig_base_ready   = eng_base.READINESS_FILE
    orig_base_health  = eng_base.HEALTH_REPORT_FILE
    orig_base_halt    = eng_base.HALT_FILE
    orig_base_force   = eng_base.FORCE_EXIT_FILE
    orig_base_log_dir = eng_base.LOG_DIR
    orig_base_log     = eng_base.LOG_FILE

    # Save engine_orders originals (stop confirmation loop reads these)
    orig_orders_timeout = eng_orders.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC
    orig_orders_poll    = eng_orders.PROTECTIVE_STOP_CONFIRM_POLL_SEC

    log_dir = str(tmp_path / "logs")
    cfg.STATE_FILE       = str(tmp_path / "engine_state.json")
    cfg.DASHBOARD_FILE   = str(tmp_path / "dashboard_data.json")
    cfg.EQUITY_HIST_FILE = str(tmp_path / "equity_history.json")
    cfg.READINESS_FILE   = str(tmp_path / "readiness_snapshot.json")
    cfg.HEALTH_REPORT_FILE = str(tmp_path / "daily_health_report.json")
    cfg.HALT_FILE        = str(tmp_path / "HALT_TRADING")
    cfg.FORCE_EXIT_FILE  = str(tmp_path / "FORCE_EXIT_ALL")
    cfg.LOG_DIR          = log_dir
    cfg.LOG_FILE         = str(tmp_path / "logs" / "trading_engine.log")

    # Patch eng (src.engine) — for any code still reading via the orchestrator
    eng.STATE_FILE       = cfg.STATE_FILE
    eng.DASHBOARD_FILE   = cfg.DASHBOARD_FILE
    eng.EQUITY_HIST_FILE = cfg.EQUITY_HIST_FILE
    eng.READINESS_FILE   = cfg.READINESS_FILE
    eng.HEALTH_REPORT_FILE = cfg.HEALTH_REPORT_FILE
    eng.HALT_FILE        = cfg.HALT_FILE
    eng.FORCE_EXIT_FILE  = cfg.FORCE_EXIT_FILE
    eng.LOG_DIR          = cfg.LOG_DIR
    eng.LOG_FILE         = cfg.LOG_FILE
    # Production waits up to 15 seconds for broker stop confirmation. Unit tests
    # use mocked IB state, so keep the same code path but shrink wall-clock wait.
    eng.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC = 0.01
    eng.PROTECTIVE_STOP_CONFIRM_POLL_SEC    = 0.01

    # Patch eng_base — where save_state / load_state / _operator_halt_active etc. read these
    eng_base.STATE_FILE       = cfg.STATE_FILE
    eng_base.DASHBOARD_FILE   = cfg.DASHBOARD_FILE
    eng_base.EQUITY_HIST_FILE = cfg.EQUITY_HIST_FILE
    eng_base.READINESS_FILE   = cfg.READINESS_FILE
    eng_base.HEALTH_REPORT_FILE = cfg.HEALTH_REPORT_FILE
    eng_base.HALT_FILE        = cfg.HALT_FILE
    eng_base.FORCE_EXIT_FILE  = cfg.FORCE_EXIT_FILE
    eng_base.LOG_DIR          = cfg.LOG_DIR
    eng_base.LOG_FILE         = cfg.LOG_FILE

    # Patch eng_orders — where _confirm_protective_stop_placed reads the timeout
    eng_orders.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC = 0.01
    eng_orders.PROTECTIVE_STOP_CONFIRM_POLL_SEC    = 0.01

    yield

    cfg.STATE_FILE       = orig_state
    cfg.DASHBOARD_FILE   = orig_dash
    cfg.EQUITY_HIST_FILE = orig_equity
    cfg.READINESS_FILE   = orig_ready
    cfg.HEALTH_REPORT_FILE = orig_health
    cfg.HALT_FILE        = orig_halt
    cfg.FORCE_EXIT_FILE  = orig_force
    cfg.LOG_DIR          = orig_log_dir
    cfg.LOG_FILE         = orig_log

    eng.STATE_FILE       = orig_state
    eng.DASHBOARD_FILE   = orig_dash
    eng.EQUITY_HIST_FILE = orig_equity
    eng.READINESS_FILE   = orig_ready
    eng.HEALTH_REPORT_FILE = orig_health
    eng.HALT_FILE        = orig_halt
    eng.FORCE_EXIT_FILE  = orig_force
    eng.LOG_DIR          = orig_log_dir
    eng.LOG_FILE         = orig_log
    eng.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC = orig_stop_confirm_timeout
    eng.PROTECTIVE_STOP_CONFIRM_POLL_SEC    = orig_stop_confirm_poll

    eng_base.STATE_FILE       = orig_base_state
    eng_base.DASHBOARD_FILE   = orig_base_dash
    eng_base.EQUITY_HIST_FILE = orig_base_equity
    eng_base.READINESS_FILE   = orig_base_ready
    eng_base.HEALTH_REPORT_FILE = orig_base_health
    eng_base.HALT_FILE        = orig_base_halt
    eng_base.FORCE_EXIT_FILE  = orig_base_force
    eng_base.LOG_DIR          = orig_base_log_dir
    eng_base.LOG_FILE         = orig_base_log

    eng_orders.PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC = orig_orders_timeout
    eng_orders.PROTECTIVE_STOP_CONFIRM_POLL_SEC    = orig_orders_poll
