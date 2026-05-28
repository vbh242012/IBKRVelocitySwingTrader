# Codex Project Memory

1. Role and project mandate

   You are an elite proprietary trader and hedge fund veteran with over 35 years of experience successfully running fully automated trading systems across equities, futures, forex, and crypto markets. You have managed nine-figure portfolios, survived multiple market crashes, and built numerous high-Sharpe, robust automated strategies that consistently generated alpha after costs, slippage, and latency.

   You also possess expert-level, up-to-date knowledge of the entire Python ecosystem, especially all the best free and open-source libraries, and constantly optimize production trading systems by leveraging the most powerful, battle-tested features from libraries such as pandas, NumPy, Numba, Polars, TA-Lib, vectorbt, Backtrader, Zipline Reloaded, QuantLib, PyAlgoTrade, ccxt, asyncio and websockets, Redis, RQ, Celery, Prometheus and Grafana, Loguru, structlog, pydantic, SQLAlchemy, Joblib, Dask, Ray, PyArrow, SciPy, Statsmodels, scikit-learn, XGBoost, LightGBM, Optuna, and others.

   We are building an automated swing trading bot with a velocity review after 1 completed trading session. For upward-trending stocks, the Chandelier trailing stop decides the exit after the velocity window if the trade is working. The bot is for a cash account with T+1 settlement, starting initial capital of $2,000, and it should use compounded account growth for further trading.

2. Repository ownership and active project folder

   - The only active project path is `/home/harika/MyLearning/AI/IBKRVelocitySwingTrader`.
   - Do not reference, compare against, read from, or modify any older/original project folder unless the user explicitly gives a new instruction in the current conversation.
   - All future code changes, tests, backtests, dashboard work, cleanup, and documentation updates must be done inside `/home/harika/MyLearning/AI/IBKRVelocitySwingTrader`.
   - Preserve user changes. Never revert unrelated work.

3. Trading-system standard

   - Be brutally honest about live-trading readiness, flaws, risk, and implementation gaps.
   - Never imply profit is guaranteed.
   - Optimize for survival first: broker state reconciliation, settled cash, order protection, kill switches, reconnection, monitoring, and realistic validation.
   - Every trading-rule change must be validated with tests and, when relevant, forward/backtest comparison.

4. Current strategy intent

   - Instrument universe: US equities through IBKR.
   - Live scanner default: broad active corporate stocks via `MOST_ACTIVE`, not `TOP_PERC_GAIN`; strategy rules decide momentum quality.
   - Account type: cash account, T+1 settlement.
   - Live capital must come from IBKR `NetLiquidation` / `SettledCash`, not a local seed constant.
   - Maximum position capacity should be calculated from total equity, capped by explicit risk settings (`VELOCITY_MIN_BUCKET_SIZE`, `VELOCITY_MAX_POSITIONS_CAP`).
   - New-entry sizing must use settled cash, not total equity.
   - Never use `AvailableFunds` as a substitute for `SettledCash` in a cash account.
   - Bucket size should be settled cash divided by remaining cash-qualified open slots.
   - Live scanner output should evaluate all unique IBKR screener results; do not cap live candidates with a fixed `SCAN_COUNT`.
   - Live commissions must come from IBKR commission reports/fill data. Keep commission constants only as backtest assumptions.
   - Entry logic should prefer liquid, high-quality momentum stocks, with special caution in bear regimes.
   - Exit logic includes Chandelier trailing stop, hard stop, Friday/weekend risk handling, velocity/stagnation exit, and emergency liquidation.
   - The VIX risk filter is mandatory for live entries. If VIX market data is missing, invalid, or above the configured threshold, the engine must skip new entries while still managing existing positions.
   - Stock entries require real-time equity market data. VIX may use delayed IBKR market data as a regime-only safety input via `VELOCITY_VIX_MARKET_DATA_TYPE=3`; the engine must restore real-time stock data mode before scanning/ordering.
   - Forward backtests must model T+1 cash settlement: sale proceeds count toward equity while unsettled but cannot fund new entries until the next trading session.
   - Current yfinance/NASDAQ-listing backtests are useful regression checks, but they are not survivorship-free institutional research. Treat strong results as provisional until validated on point-in-time historical universe data.
   - Current validated production entry rule set is Cartesian mask `8096`.
   - 8096 active entry gates are: ORB/previous-high break, gap cap, RSI minimum delta, RSI minimum level, close location in the upper part of the day range, minimum intraday gain from open, and ATR% cap.
   - Live-only fixed controls remain active around 8096: spread cap, 20-day dollar-volume floor, VIX/SPY regime handling, reduced bear-phase risk, correlation cap, sector cap, settled-cash sizing, and T+1 settlement.
   - Exhaustive validation promoted removing these old entry gates: MA50/MA200 trend, SMA200 slope, VCP/ATR contraction, near-10-day-high proximity, RVOL minimum, and plain RSI-rising. RVOL/VCP/trend may still be logged or used for scoring/ranking, but they must not reject an otherwise valid 8096 entry.

5. Validation rules

   - Keep tests synchronized with production code.
   - Run focused tests after targeted changes.
   - Run the full suite before declaring code complete:

     ```bash
     cd /home/harika/MyLearning/AI/IBKRVelocitySwingTrader
     .venv/bin/python -m pytest -q
     ```

   - For dashboard or entry-point changes, also run:

     ```bash
     .venv/bin/python -m py_compile dashboard_server.py src/engine.py src/config.py run_backtest.py
     ```

6. IBKR live/paper prerequisites

   - IB Gateway or TWS must be running with API access enabled.
   - Paper mode normally uses port `4002`.
   - Live mode normally uses port `4001` and must require explicit acknowledgement.
   - Real-time market data type must be `1`; delayed data must not drive live entries.
   - Required market data subscriptions are:
     - US Securities Snapshot and Futures Value Bundle
     - US Equity and Options Add-On Streaming Bundle
     - Cboe Streaming Market Indexes, another IBKR entitlement that provides live VIX index data through the API, or IBKR delayed VIX data through `VELOCITY_VIX_MARKET_DATA_TYPE=3`
   - NASDAQ Network C/UTP may be useful if not already covered.
   - Do not bypass the VIX filter to make the app trade without VIX data unless the user explicitly reverses this decision.

7. Runtime controls

   - `HALT_TRADING` blocks new entries but keeps managing existing positions.
   - `FORCE_EXIT_ALL` liquidates tracked positions.
   - `velocity_engine.lock` prevents duplicate engine instances.
   - Dashboard should remain independent from the trading engine and must not affect order execution.

8. Launch commands

   Paper mode:

   ```bash
   cd /home/harika/MyLearning/AI/IBKRVelocitySwingTrader
   export VELOCITY_TRADING_MODE=paper
   export VELOCITY_IB_PORT=4002
   export VELOCITY_MARKET_DATA_TYPE=1
   export VELOCITY_VIX_MARKET_DATA_TYPE=3
   .venv/bin/python auto_trader.py
   ```

   Dashboard:

   ```bash
   cd /home/harika/MyLearning/AI/IBKRVelocitySwingTrader
   .venv/bin/python dashboard_server.py --host 127.0.0.1 --port 8080
   ```

   Live mode requires:

   ```bash
   export VELOCITY_TRADING_MODE=live
   export VELOCITY_IB_PORT=4001
   export VELOCITY_MARKET_DATA_TYPE=1
   export VELOCITY_VIX_MARKET_DATA_TYPE=3
   export VELOCITY_LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_RISK
   ```

9. Code-quality rules

   - Keep changes scoped and readable.
   - Prefer existing code patterns over unnecessary new abstractions.
   - Avoid stale constants in dashboard text, tests, and backtest parameters.
   - Do not hide or ignore failed broker calls.
   - Do not allow duplicate orders, unprotected positions, or state/broker divergence to persist silently.

10. Live-safety fixes carried forward from the original project review

   - Never set IBKR `goodAfterTime` to a past timestamp. Entry bracket orders omit it after the 10:00 ET entry gate.
   - `liquidate()` must keep existing TRAIL SELL protection live until the market exit is confirmed by IBKR position sync; only non-TRAIL orders are cancelled before the market sell.
   - Liquidation market sells must be SMART-routed even if IBKR reports the position contract with a native exchange.
   - Filled liquidation attempts mark state `pending_exit=True`; state is removed only after IBKR sync confirms the position is flat.
   - `_sync_positions_from_ibkr()` must backfill `fill_price`, `broker_avg_cost`, and `peak_price` for positions recovered after a restart.
   - `_preflight_order()` must handle both `OrderState` and `[OrderState]` returns from IBKR `whatIfOrder()`.
   - Live break-even protection is dual enforced: dashboard/effective stop floors at entry after the 4% threshold, and `check_velocity_exits()` market-sells if a prior 4%+ winner retraces to entry.

11. Latest production-safety changes to preserve

   These changes were applied after the latest high-scrutiny review and must not be regressed:

   - Startup orphan cleanup now handles both orphaned `BUY` and orphaned `SELL` orders, but only when the symbol is absent from local state and absent from live IBKR positions.
   - Startup must preserve protective `SELL` orders when either local state has the symbol or IBKR reports an actual position for that symbol.
   - `liquidate()` keeps existing TRAIL protection live while the market sell is uncertain, but catches IBKR `placeOrder()` exceptions, clears `pending_exit`, retains state, and alerts so the exit can retry.
   - After IBKR confirms a symbol is flat, `_sync_positions_from_ibkr()` must cancel any leftover SELL exit orders before removing local state. This prevents orphaned trailing stops from becoming unintended future sell orders.
   - Liquidation state removal remains confirmation-based: one missing IBKR snapshot only defers removal unless `FORCE_EXIT_ALL` is active.
   - `backtest/optimizer.py` must optimize only active 8096 parameters. Legacy `rvol_min`, `breakout_pct`, and `vcp_ratio` are retained for API/report compatibility but must not be swept as if they affect entries.
   - `run_backtest.py` documentation must describe RVOL as scanner ranking only, not an entry gate.

12. Latest validation record

   Last full validation after the current safety changes:

   ```bash
   cd /home/harika/MyLearning/AI/IBKRVelocitySwingTrader
   .venv/bin/python -B -c "import ast, pathlib; files=[p for root in ['src','backtest','tests'] for p in pathlib.Path(root).rglob('*.py')] + [pathlib.Path('run_backtest.py')]; [ast.parse(p.read_text(), filename=str(p)) for p in files]; print(f'parsed {len(files)} python files')"
   PYTHONDONTWRITEBYTECODE=1 VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest tests -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --yearly --vix-delay-bars 1
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --vix-delay-bars 1
   ```

   Results:

   - Syntax parse: 16 Python files parsed cleanly.
   - Tests: 308 passed.
   - Delayed-VIX aggregate forward backtest, 2020-01-01 through 2026-05-22: 2,326 trades, +18,014.09% total return, 76.9% win rate, 9.44 profit factor, -4.17% max drawdown, 5.29 Sharpe, final equity $362,281.83 from $2,000.
   - Yearly delayed-VIX forward results:

     ```text
     2020 | 232 trades | +134.18% | Win 75.0% | PF 6.63 | MaxDD -4.17% | Sharpe 4.21
     2021 | 382 trades | +180.60% | Win 69.4% | PF 5.14 | MaxDD -3.82% | Sharpe 5.11
     2022 |  72 trades |   +9.08% | Win 66.7% | PF 3.26 | MaxDD -2.13% | Sharpe 1.38
     2023 | 266 trades | +123.45% | Win 74.1% | PF 6.38 | MaxDD -4.18% | Sharpe 5.29
     2024 | 328 trades | +131.03% | Win 70.1% | PF 3.89 | MaxDD -9.05% | Sharpe 4.25
     2025 | 326 trades | +207.89% | Win 75.5% | PF 6.62 | MaxDD -3.33% | Sharpe 6.08
     2026 | 104 trades | +120.61% | Win 78.8% | PF 20.57 | MaxDD -3.95% | Sharpe 7.73
     ```

13. Current honest readiness assessment

   - The application is safer than before, especially around broker-state reconciliation and orphan order handling.
   - It is not yet institutional-grade research infrastructure. Current backtests are regression-quality and useful, but still rely on a cached yfinance/NASDAQ-style universe and are not survivorship-free or point-in-time complete.
   - Before meaningful live allocation, require an IBKR paper-trading soak, review of every real order/fill/cancel event, alert delivery validation, dashboard authentication if exposed beyond localhost, and comparison against a point-in-time historical universe with delisted symbols.
   - For the current small cash-account plan, any real-money launch should begin only after paper validation and with the planned limited capital, not aggressive scaling.

14. Backtest entry-price realism fix from full-universe validation

   A full-universe yearly validation exposed a false 2021 max drawdown and inflated return path caused by daily-bar entry realism, especially symbols whose signal-day close was above the $20 production floor but whose simulated entry fill used a sub-$20 open/previous-high proxy. Example investigated: `GBR` on 2021-01-28, where the old daily simulator could enter near $2.43 even though the signal used the completed day close near $25. This is not live-tradable and must not return.

   Preserved fixes:

   - Backtest entries now reject candidates when the actual simulated raw entry price is below `SCAN_MIN_PRICE`, even if the signal-day close is above the scanner floor.
   - Live entry revalidation also rejects refreshed prices below `SCAN_MIN_PRICE`, so the scanner price floor is not trusted blindly after repricing.
   - `run_backtest.py` includes an optional `--conservative-daily-entry` research switch that fills no better than the completed signal-day close; this is diagnostic only and is not the production default.
   - Regression tests cover both backtest entry-price floor behavior and live reprice minimum-price validation.

   Full-universe validation after the default entry-price floor fix:

   ```text
   2020 |  727 trades | +746.75% | Win 77.4% | PF 10.01 | MaxDD -3.56% | Sharpe 7.63
   2021 |  982 trades | +723.04% | Win 78.1% | PF  8.43 | MaxDD -3.83% | Sharpe 7.87
   2022 |  353 trades |  +85.77% | Win 71.7% | PF  4.33 | MaxDD -4.09% | Sharpe 3.86
   2023 |  385 trades | +324.76% | Win 79.7% | PF 11.36 | MaxDD -7.00% | Sharpe 6.20
   2024 |  796 trades | +678.78% | Win 80.0% | PF  8.87 | MaxDD -3.49% | Sharpe 8.51
   2025 |  487 trades | +475.28% | Win 79.7% | PF 11.58 | MaxDD -6.09% | Sharpe 6.52
   2026 |  160 trades | +195.22% | Win 83.8% | PF 12.32 | MaxDD -4.17% | Sharpe 9.17
   ```

   Conservative close-or-worse daily-fill research result for 2021 was not promoted as the default: 287 trades, -47.56% return, 41.1% win rate, 0.58 profit factor, -50.21% max drawdown, -2.49 Sharpe. That mode is useful to prove how sensitive daily-bar research is to fill assumptions, but it is too pessimistic compared with the live intraday ORB process.

   Validation after code/test updates: `317 passed in 3.86s`.

15. MOST_ACTIVE live entry pricing rule

   The live engine no longer prices BUY entries as a blind fixed `price * 1.002`
   limit. That 0.2% value is now only the hard maximum overpay cap. For
   `MOST_ACTIVE` scanning, parent BUY orders use spread-aware ask-based
   marketable limits.

   - Technical context must carry `bid`, `ask`, and `spread_pct`.
   - Entry still requires `spread_pct <= SPREAD_MAX_PCT`.
   - The parent BUY limit is calculated from the validated ask plus a small
     cushion: `max(ENTRY_LIMIT_MIN_TICK, ask * ENTRY_LIMIT_ASK_CUSHION_PCT)`.
   - The resulting limit is capped by
     `price * (1 + ENTRY_LIMIT_MAX_OVER_MARKET_PCT)`, where the default max cap
     is still 0.2%.
   - If bid/ask are missing, crossed, too wide, or ask is already above the max
     cap, the trade is skipped rather than using a stale or blind price.
   - Stale scan prices must re-fetch price, bid, and ask before order placement;
     if fresh bid/ask are unavailable, skip.

   Defaults:

   ```text
   ENTRY_LIMIT_ASK_CUSHION_PCT = 0.0005
   ENTRY_LIMIT_MIN_TICK = 0.01
   ENTRY_LIMIT_MAX_OVER_MARKET_PCT = 0.002
   ```

   Validation after the change:

   - Focused pricing/stale-reprice tests: 5 passed.
   - Full suite: 320 passed in 4.04s.
   - Full-universe yearly backtest unchanged from the entry-price-floor
     validation, confirming no research-path regression.

16. Optional IB Gateway / IBC auto-start integration

   The trading app can optionally supervise an external IB Gateway or IBC
   launcher before connecting to IBKR. The application must not store broker
   credentials or bypass IBKR two-factor authentication. Credentials, if used,
   belong in external IBC configuration or an OS secret mechanism.

   Environment variables:

   ```text
   VELOCITY_IB_GATEWAY_AUTO_START=1
   VELOCITY_IB_GATEWAY_START_CMD="/path/to/your/ibc-start-script.sh"
   VELOCITY_IB_GATEWAY_START_TIMEOUT_SEC=180
   VELOCITY_IB_GATEWAY_START_POLL_SEC=2
   VELOCITY_IB_GATEWAY_STOP_ON_EXIT=0
   VELOCITY_IB_GATEWAY_LOG_FILE=/path/to/ib_gateway_launcher.log
   ```

   Behavior:

   - `VelocityEngine.connect()` checks whether the configured IB API port is
     reachable before calling `IB.connect()`.
   - If the port is closed and auto-start is disabled, the engine fails closed
     and does not try to trade.
   - If auto-start is enabled, the app starts the external command with
     `subprocess.Popen()` using `shlex.split()` and no shell.
   - The app waits until `VELOCITY_IB_HOST:VELOCITY_IB_PORT` accepts socket
     connections.
   - Reconnect attempts also call the same readiness function, so a dead
     Gateway can be relaunched by the configured external supervisor.
   - Launcher stdout/stderr are written to `VELOCITY_IB_GATEWAY_LOG_FILE`.

   Validation after the change:

   - Gateway/connection focused tests: 9 passed.
   - Full suite: 327 passed in 3.67s.
   - Full-universe yearly backtest unchanged from the latest validated strategy
     results.

17. Runtime scan/audit safety pass

   Preserve these live-engine safety changes:

   - `_daily_scan_skip` exists only as an IBKR pacing guard. It caches stable
     same-day scan failures, not dynamic intraday failures.
   - Cached same-day scan failures currently include insufficient daily history,
     invalid daily MA200, and 20-day dollar volume below the active threshold.
   - Dynamic failures such as ORB not yet broken, wide spread, gap too high,
     weak day-location, weak open-gain, or ATR% too high must not be cached.
   - Runtime cycles now run a protective stop audit once per trading day and
     immediately whenever local state suggests an open position has no valid
     stop distance or stop loss.
   - The runtime stop audit runs before account/VIX/new-entry gates, so existing
     positions can still be protected even when account summary or regime data
     is temporarily unavailable.
   - VIX ticker request exceptions now fall back to the historical VIX lookup
     instead of bypassing the fallback path.

   Validation after this pass:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -p no:cacheprovider -q
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c "import ast, pathlib; files=[p for root in ['src','backtest','tests'] for p in pathlib.Path(root).rglob('*.py')] + [pathlib.Path('run_backtest.py'), pathlib.Path('auto_trader.py'), pathlib.Path('dashboard_server.py')]; [ast.parse(p.read_text(), filename=str(p)) for p in files]; print(f'parsed {len(files)} python files')"
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --yearly --vix-delay-bars 1
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --vix-delay-bars 1
   ```

   Results:

   - Tests: 337 passed.
   - Syntax parse: 20 Python files parsed cleanly.
   - Cached 282-symbol aggregate forward backtest, 2020-01-01 through
     2026-05-22: 2,585 trades, +1,821.57% total return, 73.5% win rate,
     6.16 profit factor, -4.14% max drawdown, 4.26 Sharpe, final equity
     $38,431.37 from $2,000.
   - Cached 282-symbol yearly delayed-VIX forward results:

     ```text
     2020 | 222 trades | +139.90% | Win 75.2% | PF  7.33 | MaxDD -4.14% | Sharpe 4.25
     2021 | 362 trades | +175.49% | Win 69.3% | PF  5.10 | MaxDD -4.00% | Sharpe 5.11
     2022 |  69 trades |  +10.95% | Win 66.7% | PF  3.65 | MaxDD -2.09% | Sharpe 1.46
     2023 | 252 trades | +120.00% | Win 73.8% | PF  6.44 | MaxDD -4.08% | Sharpe 5.13
     2024 | 317 trades | +131.79% | Win 69.4% | PF  3.91 | MaxDD -8.86% | Sharpe 4.30
     2025 | 320 trades | +217.08% | Win 75.6% | PF  7.98 | MaxDD -3.34% | Sharpe 6.16
     2026 | 103 trades | +125.10% | Win 79.6% | PF 22.92 | MaxDD -3.94% | Sharpe 7.63
     ```

   Cleanup after validation:

   - Removed generated Python cache folders: `src/__pycache__`,
     `tests/__pycache__`, `backtest/__pycache__`, project `__pycache__`, and
     `.pytest_cache`.
   - Left `backtest/.cache/` and `logs/` intact because they are validation and
     runtime evidence, not obsolete code.

18. Paper trading soak / IBC launcher setup

   Current active path remains:
   `/home/harika/MyLearning/AI/IBKRVelocitySwingTrader`

   The machine has IB Gateway installed at:
   `/home/harika/Jts/ibgateway/1046/ibgateway`

   IBC 3.23.0 was installed locally at:
   `/home/harika/ibc`

   The initial IBC startup/config files were adjusted for paper trading, then
   superseded by the profile-specific setup in section 19:

   - `/home/harika/ibc/gatewaystart.sh`
     - `TWS_MAJOR_VRSN=1046`
     - `TRADING_MODE=paper`
     - `TWOFA_TIMEOUT_ACTION=restart`
     - `IBC_PATH=/home/harika/ibc`
   - `/home/harika/ibc/config.ini`
     - blanked sample demo credentials
     - `TradingMode=paper`
     - `AcceptNonBrokerageAccountWarning=yes`
     - `ExistingSessionDetectedAction=primary`
     - `OverrideTwsApiPort=4002`
     - `ReadOnlyApi=no`
     - `AutoRestartTime=17:00`
     - `AcceptIncomingConnectionAction=accept`

   The IBC folder was restricted to the local Linux user (`700`), and the IBC
   config files were restricted to owner read/write (`600`). Add credentials
   only in the profile-specific local IBC config files listed in section 19 or
   an OS secret mechanism. Do not put IBKR credentials in this repo.

   Added local run helpers. These were later simplified in section 20 so only
   the profile-aware scripts remain active.

   - `.env.paper.example`
   - `.env.paper.local` (ignored by git, already created)
   - `scripts/start_dashboard.sh`
   - `scripts/start_trader.sh`
   - `scripts/check_runtime.sh`

   The intended paper configuration is:

   ```text
   VELOCITY_TRADING_MODE=paper
   VELOCITY_IB_HOST=127.0.0.1
   VELOCITY_IB_PORT=4002
   VELOCITY_IB_CLIENT_ID=1
   VELOCITY_MARKET_DATA_TYPE=1
   VELOCITY_VIX_MARKET_DATA_TYPE=3
   VELOCITY_IB_GATEWAY_AUTO_START=1
   VELOCITY_BASE_DIR=/home/harika/MyLearning/AI/IBKRVelocitySwingTrader/runtime/paper
   VELOCITY_IB_GATEWAY_START_CMD="/home/harika/ibc/start-paper-gateway.sh"
   VELOCITY_IB_GATEWAY_START_TIMEOUT_SEC=240
   VELOCITY_IB_GATEWAY_START_POLL_SEC=2
   VELOCITY_IB_GATEWAY_STOP_ON_EXIT=0
   VELOCITY_IB_GATEWAY_LOG_FILE=runtime/paper/logs/ib_gateway_launcher.log
   ```

   Run commands:

   ```bash
   nohup ./scripts/start_trader.sh paper > logs/paper_autotrader_stdout.log 2> logs/paper_autotrader_stderr.log &
   nohup ./scripts/start_dashboard.sh paper > logs/paper_dashboard_stdout.log 2> logs/paper_dashboard_stderr.log &
   ./scripts/check_runtime.sh paper
   ```

   Operational warning: a 15-20 day unattended paper run is reasonable only
   after the first few starts prove that IBC handles login, Gateway restart,
   IBKR maintenance windows, market-data reconnection, and any two-factor prompt
   behavior for this account. Until webhook alerts are configured and observed,
   check the process/logs daily rather than waiting several days.

19. Paper/live profile switching

   The app is now switchable by profile instead of source-code edits.

   Repository-side files:

   - `.env.paper.example`
   - `.env.live.example`
   - `.env.paper.local` (ignored by git)
   - `.env.live.local` (ignored by git; live ACK remains commented)
   - `scripts/start_trader.sh [paper|live]`
   - `scripts/start_dashboard.sh [paper|live]`
   - `scripts/check_runtime.sh [paper|live]`

   Runtime isolation:

   - paper: `runtime/paper`
   - live: `runtime/live`

   This prevents paper `engine_state.json`, `dashboard_data.json`,
   `equity_history.json`, lock files, and logs from mixing with live mode.

   IBC-side files:

   - `/home/harika/ibc/config.paper.ini`
   - `/home/harika/ibc/config.live.ini`
   - `/home/harika/ibc/start-paper-gateway.sh`
   - `/home/harika/ibc/start-live-gateway.sh`

   IBC profile mapping:

   ```text
   paper app profile -> IB API port 4002 -> /home/harika/ibc/config.paper.ini
   live app profile  -> IB API port 4001 -> /home/harika/ibc/config.live.ini
   ```

   Live trading is blocked by two gates:

   - `.env.live.local` must set `VELOCITY_TRADING_MODE=live`.
   - `.env.live.local` must explicitly set
     `VELOCITY_LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_RISK`.

   Run commands:

   ```bash
   nohup ./scripts/start_trader.sh paper > logs/paper_autotrader_stdout.log 2> logs/paper_autotrader_stderr.log &
   nohup ./scripts/start_dashboard.sh paper > logs/paper_dashboard_stdout.log 2> logs/paper_dashboard_stderr.log &
   ./scripts/check_runtime.sh paper

   nohup ./scripts/start_trader.sh live > logs/live_autotrader_stdout.log 2> logs/live_autotrader_stderr.log &
   nohup ./scripts/start_dashboard.sh live > logs/live_dashboard_stdout.log 2> logs/live_dashboard_stderr.log &
   ./scripts/check_runtime.sh live
   ```

   Do not uncomment the live ACK until paper trading has run cleanly and
   `/home/harika/ibc/config.live.ini` is confirmed to log into the live account
   on port `4001`.

20. Operator script cleanup

   Removed redundant compatibility wrappers for paper-only/live-only trader
   startup and paper-only runtime checks.

   Active operator scripts are now intentionally limited to:

   - `scripts/start_trader.sh [paper|live]`
   - `scripts/start_dashboard.sh [paper|live]`
   - `scripts/check_runtime.sh [paper|live]`

   The active shell scripts include comments explaining profile selection,
   local env loading, runtime isolation, live-mode acknowledgement, and health
   checks. This keeps the operating surface small and reduces the chance of
   starting the wrong account mode.

21. Live connectivity pre-market fixes, 2026-05-28

   A live pre-market startup test against IB Gateway on port `4001` found and
   fixed three broker-state edge cases:

   - Stale local positions that are missing from one IBKR position snapshot are
     no longer refreshed with historical/market-data calls while awaiting the
     second confirming snapshot. This prevented stale `DELL` state from causing
     historical data timeouts.
   - Startup now runs an immediate confirmation sync and protective-stop audit
     when local state contains positions with missing/zero stop fields instead
     of waiting until the 09:58 ET pre-entry sync.
   - Stop audits now request broker-wide open orders via `reqAllOpenOrders()`
     before using `openTrades()`. This prevents the engine from missing
     already-live GTC protective orders and trying to create duplicate sell
     stops in a cash account.

   IBKR can represent an existing TRAIL sell as a percentage trail where
   `auxPrice` is `UNSET_DOUBLE`, while the real protection is carried in
   `trailStopPrice` and `trailingPercent`. The audit logic now recognizes both:

   - Dollar TRAIL: finite positive `auxPrice`.
   - Percent TRAIL: finite positive `trailStopPrice` plus `trailingPercent`.

   Runtime state now records percent-trail stops as:

   ```text
   stop_mode = "percent"
   trailing_percent = <IBKR trailingPercent>
   stop_loss/effective_stop = <IBKR trailStopPrice snapshot>
   ```

   The dashboard no longer invents a fixed-dollar effective stop for percent
   trailing orders.

   Live validation result:

   - IB Gateway live API port `127.0.0.1:4001` connected successfully.
   - Dashboard live API on `127.0.0.1:8081` served connected live runtime state.
   - Existing protective GTC TRAIL sells confirmed:
     - `SBUX`: qty 4, stop `$98.92`, trail `4.807%`
     - `OXY`: qty 10, stop `$55.22`, trail `6.5614%`
     - `CSCO`: qty 4, stop `$114.05`, trail `5.5749%`
   - Live autotrader reached the expected pre-entry wait state:
     `Waiting until 09:58 ET for pre-entry position sync & stop audit`.

   Validation after the code changes:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest tests/test_startup_init.py -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   ```

   Results:

   - Focused startup/audit tests: 70 passed.
   - Full suite: 344 passed.

22. Protective TRAIL activation gate, 2026-05-28

   Requirement: protective TRAIL sell orders must not activate before 09:32 ET.
   This is deliberately separate from the 10:00 ET new-entry gate: existing
   positions get protection after the first 2 opening minutes, while new BUY
   entries still wait until 10:00 ET.

   Implementation details to preserve:

   - `STOP_ACTIVATION_TIME = (9, 32)` controls protective TRAIL stop
     activation.
   - New pre-09:32 audit-created TRAIL orders set
     `goodAfterTime=YYYYMMDD 09:32:00 US/Eastern`.
   - New entry bracket parent BUY and child TRAIL orders share the same
     `goodAfterTime` value when submitted before 10:00 ET; after 10:00 ET the
     field is omitted because IBKR rejects past activation timestamps.
   - Existing GTC TRAIL orders recovered from IBKR are checked during stop
     audit. If they lack the current day's 09:32 ET `goodAfterTime`, the engine
     cancels and replaces them with equivalent TRAIL orders carrying the gate.
   - The engine must use the same IB API `clientId` that owns the existing
     orders. Orders from a different `clientId` are visible through
     `reqAllOpenOrders()` but may not be cancellable/modifiable. In that case
     the engine alerts and does not attempt blind control.

   Live finding:

   - Existing SBUX/OXY/CSCO protective stops were owned by IB API `clientId=1`,
     while `.env.live.local` had been set to `VELOCITY_IB_CLIENT_ID=11`.
   - This mismatch caused earlier modification/cancel attempts to fail with
     duplicate/not-found behavior while `reqAllOpenOrders()` still displayed
     the orders.
   - `.env.live.local` and `.env.live.example` were aligned to
     `VELOCITY_IB_CLIENT_ID=1`.

   Live order result after applying the fix:

   ```text
   SBUX | order 7256 | clientId 1 | qty 4  | goodAfterTime 20260528 10:00:00 US/Eastern | stop 98.92  | trail 4.807%
   OXY  | order 7257 | clientId 1 | qty 10 | goodAfterTime 20260528 10:00:00 US/Eastern | stop 55.22  | trail 6.5614%
   CSCO | order 7258 | clientId 1 | qty 4  | goodAfterTime 20260528 10:00:00 US/Eastern | stop 114.05 | trail 5.5749%
   ```

   Live autotrader was restarted with `clientId=1` and reached the normal
   pre-entry wait state.

   Validation after the change:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest tests/test_startup_init.py -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py
   ```

   Results:

   - Focused startup/audit tests: 72 passed.
   - Full suite: 346 passed.
   - `py_compile`: passed.

   Updated stop activation request:

   - User changed the desired protective stop activation from 10:00 ET to
     09:32 ET.
   - Code now uses `_stop_good_after_time()` for protective stop audits and
     keeps `_entry_good_after_time()` for new entry brackets.
   - Existing live TRAIL orders were re-audited/replaced. Because the change was
     applied at 11:42 ET, after the 09:32 ET stop gate had already passed, the
     broker-side orders were replaced with blank `goodAfterTime`, meaning active
     immediately. Future pre-09:32 audits will use `09:32:00 US/Eastern`.
   - If an old order still has a later future `goodAfterTime` after the stop
     gate has already passed, the audit cancels/replaces it with an immediately
     active equivalent TRAIL order.

   Live order result after the 09:32 update:

   ```text
   SBUX | order 7261 | clientId 1 | qty 4  | goodAfterTime '' | stop 98.92  | trail 4.807%
   OXY  | order 7262 | clientId 1 | qty 10 | goodAfterTime '' | stop 55.22  | trail 6.5614%
   CSCO | order 7263 | clientId 1 | qty 4  | goodAfterTime '' | stop 114.05 | trail 5.5749%
   ```

   Restart validation exposed a separate live exit-order issue:

   - Velocity exits attempted to liquidate SBUX/OXY/CSCO after the hold window.
   - IBKR rejected the market sells because presets changed the market order TIF
     to GTC, producing `Invalid effective time`.
   - Liquidation market sells now explicitly set `tif='DAY'` and
     `goodAfterTime=''` so IBKR presets should not convert them to invalid GTC
     market orders.
   - The live autotrader was stopped after the rejection and should not be
     restarted without understanding that it may immediately retry those
     velocity exits and sell the positions.

   Latest validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest tests/test_startup_init.py tests/test_trailing_stop_scoring_screener.py -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py
   ```

   Results:

   - Focused startup/trailing-stop tests: 236 passed.
   - Full suite: 347 passed.
   - `py_compile`: passed.

23. Live cash-account exit and entry preflight findings, 2026-05-28

   Live restart was intentionally allowed to execute strategy rules. It exposed
   two real IBKR cash-account behaviors that mocks/backtests did not catch:

   - A full-quantity protective SELL order and a full-quantity market SELL exit
     cannot coexist in a cash account. IBKR rejects the market exit as a
     potential short sale.
   - IBKR what-if validation requires `transmit=True`, even when the real
     bracket parent BUY must remain `transmit=False` until the child stop is
     attached.

   Fixes applied:

   - `liquidate()` now cancels all active open orders for the symbol, waits for
     active SELL orders to clear, then submits the `DAY` market SELL. If the
     market SELL placement/rejection path fails after protection was cancelled,
     the engine immediately runs the stop audit to rebuild protection.
   - `_preflight_order()` now validates a copied order with `whatIf=True` and
     `transmit=True`; the live order object is not mutated.

   Live validation:

   - SBUX/OXY/CSCO velocity exits first failed with cash-account short-sale
     rejection while stops were live.
   - After the cash-account exit fix, the engine cancelled each protective stop
     and sold all three positions successfully:
     - SBUX: market SELL filled, qty 4.
     - OXY: market SELL filled, qty 10.
     - CSCO: market SELL filled, qty 4.
   - Broker verification after exits showed no stock positions and no open
     orders.
   - Local state reconciled to `{}` after the two-snapshot missing-position
     guard.
   - After the preflight fix, the engine entered RIOT:
     - RIOT BUY filled, qty 21, fill about $28.30.
     - Commission report captured: about $1.0001.
     - Broker verification showed RIOT position qty 21 and live TRAIL SELL
       protection order 7546, qty 21, trail distance $2.86.
     - The next stop audit confirmed RIOT TRAIL SELL live.

   Operational note:

   - In this execution environment, plain `nohup ./scripts/start_trader.sh live
     ... &` returned but did not keep the trader process alive. `setsid -f
     ./scripts/start_trader.sh live > logs/live_autotrader_stdout.log 2>
     logs/live_autotrader_stderr.log < /dev/null` did keep it alive.
   - IB Gateway was already running, so app-side `ensure_ib_gateway_ready()`
     returned `True` without launching IBC. The live IBC launcher file exists
     and is executable, but a full auto-login test would require deliberately
     stopping Gateway and should not be done casually while live trading is
     active.

   Latest validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest tests/test_startup_init.py::TestPreflightOrder tests/test_trailing_stop_scoring_screener.py::TestExitOrders -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py
   ```

   Results:

   - Focused preflight/liquidation tests: 23 passed.
   - Full suite: 348 passed.
   - `py_compile`: passed.
