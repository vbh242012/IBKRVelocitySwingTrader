# Codex Project Memory

1. Role and project mandate

   You are an elite proprietary trader and hedge fund veteran with over 35 years of experience successfully running fully automated trading systems across equities, futures, forex, and crypto markets. You have managed nine-figure portfolios, survived multiple market crashes, and built numerous high-Sharpe, robust automated strategies that consistently generated alpha after costs, slippage, and latency.

   You also possess expert-level, up-to-date knowledge of the entire Python ecosystem, especially all the best free and open-source libraries, and constantly optimize production trading systems by leveraging the most powerful, battle-tested features from libraries such as pandas, NumPy, Numba, Polars, TA-Lib, vectorbt, Backtrader, Zipline Reloaded, QuantLib, PyAlgoTrade, ccxt, asyncio and websockets, Redis, RQ, Celery, Prometheus and Grafana, Loguru, structlog, pydantic, SQLAlchemy, Joblib, Dask, Ray, PyArrow, SciPy, Statsmodels, scikit-learn, XGBoost, LightGBM, Optuna, and others.

   We are building an automated swing trading bot where broker-side Chandelier trailing stops are the primary exit for working trades. The only maintained strategy profile is `indicator_swing`: a relative-strength-first swing system, not an intraday ORB system and not a long-term investment system. The bot is for a cash account with T+1 settlement, starting initial capital of $2,000, and it should use compounded account growth for further trading.

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
   - Default and only maintained profile: `indicator_swing`.
   - Removed profiles/code paths must not be reintroduced casually. Old ORB/current, standalone reversal/momentum, standalone Bollinger, standalone PSAR, legacy/enhanced/swing scoring models, and old backtest compatibility knobs were removed on 2026-06-11 to keep live code, tests, dashboard, and backtester aligned.
   - Current production intent: relative-strength-first swing momentum. A stock must first pass trend/leadership gates before any indicator timing sleeve can buy.
   - Live application scanner default: `VELOCITY_APP_SCANNER_SOURCE=hybrid`, meaning curated IBKR scanner hits plus a rotating full US common-stock universe batch. Strategy rules decide final momentum quality.
   - Premarket scanner architecture: at `VELOCITY_APP_PREFILTER_START_TIME` (default `06:30 ET`), scan the full configured common-stock universe once, apply only static historical filters that cannot become true intraday, cache the surviving candidate list, and use that list for entry-window live screening. Do not spend the entry window rediscovering the full universe.
   - Startup must not block until the 09:15 ET pre-entry sync. It should enter the main loop after the immediate safety audit so the 06:30 ET prefilter can run; the 09:15 ET sync/stop audit is a scheduled checkpoint.
   - If IBKR historical pacing is too slow to finish before `ENTRY_START`, keep `VELOCITY_APP_PREFILTER_STOP_AT_ENTRY_START=1`: save a partial same-day prefilter cache with `stopped_reason=entry_window_open` and trade only from the screened subset. Manual diagnostics can use `scripts/run_premarket_prefilter.py --ignore-entry-cutoff`.
   - Account type: cash account, T+1 settlement.
   - Live capital must come from IBKR `NetLiquidation` / `SettledCash`, not a local seed constant.
   - Maximum position capacity should be calculated from total equity, capped by explicit risk settings (`VELOCITY_MIN_BUCKET_SIZE`, `VELOCITY_MAX_POSITIONS_CAP`).
   - New-entry sizing must use settled cash, not total equity.
   - Never use `AvailableFunds` as a substitute for `SettledCash` in a cash account.
   - Bucket size should be settled cash divided by remaining cash-qualified open slots.
   - Live scanner output should evaluate all unique app-scanner candidates; do not cap live candidates with a backtest-style fixed `SCAN_COUNT`. Use `VELOCITY_APP_SCANNER_BATCH_SIZE` only as the fallback pacing control before a same-day premarket prefilter cache exists.
   - Where IBKR exposes a direct scanner-side filter equivalent, apply it upstream and keep the local screener check as the final authority. Scanner-side filters are only noise reduction; the local profile evaluator is the final trading rule.
   - Live commissions must come from IBKR commission reports/fill data. Keep commission constants only as backtest assumptions.
   - Scanner price floor for the default profile is `$10`; entry logic should prefer liquid, high-quality momentum stocks, with special caution in bear regimes.
   - Default profile liquidity floors are at least 1,000,000 shares/day, $75M 20-day average dollar volume, and $1B market cap unless explicitly overridden.
   - Default hard gates include weekly uptrend, positive 3/6-month relative strength versus SPY, positive 13/26-week absolute return, price near its 52-week high, price above MA50, MA50 above MA200, rising SMA200, controlled ATR%, controlled spread, and controlled MA20 extension.
   - Default timing sleeves:
     - `ma_cross`: default live sleeve. EMA20 must be above SMA50; fresh crosses, MA20/MA50 reclaims, or prior-high breaks can time entries only after the RS/trend gate passes.
     - `bollinger_reversion`: standalone research profile only unless explicitly enabled. It means Bollinger lower-band reclaim after two prior closes below the lower band. Do not buy merely because price closed below the lower band.
     - `psar_flip`: standalone research profile only unless explicitly enabled. PSAR may be confirmation/trailing evidence, but it is not a default primary buy trigger.
   - Default confirmation rule (updated 2026-07-10): RSI must show momentum/recovery and at least two of MACD, OBV, PSAR, or stochastic must confirm. Volume pace is a separate hard gate and no longer counts toward the confirmation total — counting it double-counted the gate and collapsed the rule to one-of-four in practice.
   - MA-reclaim (pullback) `ma_cross` entries receive a bounded +5 scoring bonus (`VELOCITY_RECLAIM_TRIGGER_BONUS`, promoted 2026-07-11; 0 disables). Trigger attribution measured reclaim entries at PF 1.83 / MaxDD -6.3% vs breakout entries at PF 1.25 / MaxDD -17.3%, so reclaims win slot competition. The 0/5/10/15 sweep saturated at 5 points: bounded forward backtest improved from +39.75% to +48.58% (PF 1.29→1.35, Sharpe 0.52→0.60, MaxDD unchanged). The `VELOCITY_MA_CROSS_TRIGGERS` research switch (fresh_cross | break_prev_high | reclaim) exists for re-running the attribution; the fresh EMA20/SMA50 cross trigger was measured structurally dead inside the RS maturity gates (8 trades in 6.4 years, PF 0.28).
   - Analyst ratings are bounded scoring/exit inputs only. Analyst consensus may improve or reduce rank, but it must never create a buy by itself or force an exit without weak price action confirming the downgrade.
   - Live/paper analyst ratings resolve in this order: dated local CSV, Finnhub when `VELOCITY_FINNHUB_API_KEY` is set, then Yahoo/yfinance when `VELOCITY_ANALYST_RATINGS_FREE_SOURCE=yahoo` is enabled. Backtests use only dated local CSV snapshots to avoid look-ahead.
   - Default minimum entry score is 50.
   - Exit logic: Flat percent trailing stop (primary, broker-side IBKR TRAIL order using `trailingPercent`), hard stop, analyst downgrade exit with price confirmation, matching-sleeve exit, swing time stop, and emergency liquidation. Tiered profit exits and break-even exit were removed on 2026-06-20. Chandelier ATR-based stop was replaced by percent trail on 2026-06-24. Do not reintroduce any of them.
   - `TRAIL_PCT` (flat percent trailing stop from peak price, env `VELOCITY_TRAIL_PCT`): code default is `0.02` since 2026-06-28, but that cut from 4% to 2% was never backtest-validated. The 2026-07-10 walk-forward optimizer run (train 2020-2025, forward 2025-2026-05, bounded cached universe, clean entry logic) ranked 5% best (only setting positive in both windows; forward +19.1%, Sharpe 1.07, MaxDD -5.8%), 4% roughly break-even, and 2% worst in every column. Recommendation on record: move live to `0.05`. `CHANDELIER_PERIOD = 22` is retained only for the ATR volatility entry filter (ATR_CHAND / price for ATR_PCT_MAX gate). Optimizer grid is `[0.03, 0.04, 0.05, 0.06]` (quick: `[0.04, 0.05]`).
   - Default swing time stop is 10 trading bars when the position is not above breakeven. The maintained profile disables same-day EOD churn and Friday cleanup by default.
   - The VIX risk filter is mandatory for live entries. If VIX market data is missing, invalid, or above the configured threshold, the engine must skip new entries while still managing existing positions.
   - Stock entries require real-time equity market data. VIX may use delayed IBKR market data as a regime-only safety input via `VELOCITY_VIX_MARKET_DATA_TYPE=3`; the engine must restore real-time stock data mode before scanning/ordering.
   - Forward backtests must model T+1 cash settlement: sale proceeds count toward equity while unsettled but cannot fund new entries until the next trading session.
   - Current yfinance/NASDAQ-listing backtests are useful regression checks, but they are not survivorship-free institutional research. Treat strong results as provisional until validated on point-in-time historical universe data.
   - The old ORB/8096 rule set and `current` legacy profile were removed. Do not preserve tests, docs, config, scoring branches, or backtest parameters for them.
   - Live-only fixed controls remain active around all profiles: spread cap, 20-day dollar-volume floor, VIX/SPY regime handling, correlation cap, sector cap, settled-cash sizing, and T+1 settlement.
   - Do not add weak standalone indicator triggers to the default profile without multi-year forward validation. The previous naive Bollinger and PSAR primary-entry variants failed multi-year testing, and the default profile was improved by disabling Bollinger from the combined live sleeve set.

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

   - Never set IBKR `goodAfterTime` to a past timestamp. Entry BUY orders omit it after the configured entry gate, currently 09:45 ET.
   - In an IBKR cash account, `liquidate()` cancels active SELL orders, including protective TRAIL orders, before submitting the market exit. IBKR can otherwise reject the exit as a potential oversell because another full-size SELL is already live. If the market exit is rejected or placement fails, state is retained, `pending_exit` is cleared, an alert is emitted, and `_audit_stop_orders()` rebuilds protection immediately.
   - Liquidation market sells must be SMART-routed even if IBKR reports the position contract with a native exchange.
   - Filled liquidation attempts mark state `pending_exit=True`; state is removed only after IBKR sync confirms the position is flat.
   - `_sync_positions_from_ibkr()` must backfill `fill_price`, `broker_avg_cost`, and `peak_price` for positions recovered after a restart.
   - `_preflight_order()` must handle both `OrderState` and `[OrderState]` returns from IBKR `whatIfOrder()`.
   - Break-even exit and tiered profit exits were removed on 2026-06-20. The Chandelier broker TRAIL stop is now the sole profit-protection mechanism.

11. Latest production-safety changes to preserve

   These changes were applied after the latest high-scrutiny review and must not be regressed:

   - Startup orphan cleanup now handles both orphaned `BUY` and orphaned `SELL` orders, but only when the symbol is absent from local state and absent from live IBKR positions.
   - Startup must preserve protective `SELL` orders when either local state has the symbol or IBKR reports an actual position for that symbol.
   - `liquidate()` cancels active SELL protection before a cash-account market exit, then catches IBKR `placeOrder()` exceptions, clears `pending_exit`, retains state, alerts, and runs `_audit_stop_orders()` so protection is rebuilt if the exit cannot be placed or is rejected.
   - After IBKR confirms a symbol is flat, `_sync_positions_from_ibkr()` must cancel any leftover SELL exit orders before removing local state. This prevents orphaned trailing stops from becoming unintended future sell orders.
   - Liquidation state removal remains confirmation-based: one missing IBKR snapshot only defers removal unless `FORCE_EXIT_ALL` is active.
   - `backtest/optimizer.py` must optimize only the active `indicator_swing` exit parameter: `trail_pct`. Chandelier ATR mult was replaced by `trail_pct` on 2026-06-24.
   - `run_backtest.py` must expose only current live/backtest strategy controls.

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

16. Latest live-loop safety and health-report changes

   The latest production-safety pass added operational observability and tighter
   new-entry sequencing without changing the backtest strategy rules.

   - Existing-position management now runs before entry-only gates such as VIX,
     SPY regime, scanner calls, and available-slot checks. A bad VIX/scanner
     path must not delay software exits for positions already held.
   - VIX and scanner calls are skipped when there are no settled-cash entry
     slots, and Friday new entries are blocked after `FRIDAY_ENTRY_CUTOFF_TIME`
     while existing positions continue to be managed.
   - Filled BUY orders now require protective TRAIL confirmation. If the stop is
     not visible after the immediate audit/confirmation window, the position is
     marked `protection_status=unconfirmed`, a CRITICAL alert is emitted, and no
     further entries are attempted in that cycle.
   - Stop audits mark protection state as confirmed or unconfirmed in
     `engine_state.json` so restarts and dashboard review can see whether each
     live position is protected.
   - A compact daily operations report is written to
     `daily_health_report.json`, including cycles, IB errors, reconnects, VIX
     fallback counts, scanner counts, alert counts, protection status, equity,
     settled cash, and current tracked positions.
   - `_fetch_vix_price()` fails closed after the VIX cache TTL expires: if both
     fresh ticker data and historical fallback fail, it returns `None` and new
     entries are blocked. A stale VIX value may be reused only while it is still
     inside the 5-minute TTL.

   Validation after these changes:

   ```bash
   cd /home/harika/MyLearning/AI/IBKRVelocitySwingTrader
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py tests/conftest.py tests/test_engine.py tests/test_trailing_stop_scoring_screener.py
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-31 --yearly
   ```

   Results:

   - Syntax compile: passed.
   - Full test suite: 376 passed in 11.13s.
   - Full-universe yearly forward validation, 2020-01-01 through 2026-05-31:

     ```text
     2020 |  731 trades | +825.15% | Win 77.7% | PF 11.00 | MaxDD -4.62% | Sharpe  7.59
     2021 | 1095 trades | +941.01% | Win 79.4% | PF  9.32 | MaxDD -3.97% | Sharpe  8.51
     2022 |  358 trades |  +90.95% | Win 71.5% | PF  4.22 | MaxDD -4.88% | Sharpe  3.95
     2023 |  416 trades | +348.26% | Win 81.7% | PF 11.20 | MaxDD -6.88% | Sharpe  6.20
     2024 | 1047 trades | +999.47% | Win 81.1% | PF  9.51 | MaxDD -4.23% | Sharpe  9.25
     2025 |  493 trades | +498.89% | Win 81.5% | PF 11.40 | MaxDD -7.37% | Sharpe  6.61
     2026 |  173 trades | +315.84% | Win 84.4% | PF 21.75 | MaxDD -3.81% | Sharpe 10.97
     ```

   Follow-up validation after the VIX stale-cache fail-closed patch:

   - Regression added:
     `test_expired_stale_vix_cache_does_not_authorize_entries`.
   - Focused VIX tests: 3 passed.
   - Syntax compile with isolated pycache: passed.
   - Full test suite: 377 passed in 7.08s.
   - Cached bounded yearly backtest smoke, 2020-01-01 through 2026-05-22
     with `--max-symbols 300 --vix-delay-bars 1`: matched the prior profile.

17. Optional IB Gateway / IBC auto-start integration

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

18. Runtime scan/audit safety pass

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
     for any open position instead of waiting until the 09:15 ET pre-entry sync.
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
   - Live autotrader should finish startup and enter the main loop promptly;
     the 09:15 ET pre-entry position sync and stop audit is now a scheduled
     checkpoint, not a blocking startup wait.

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
   This is deliberately separate from the 09:45 ET new-entry gate: existing
   positions get protection after the first 2 opening minutes, while new BUY
   entries still wait until 09:45 ET.

   Implementation details to preserve:

   - `STOP_ACTIVATION_TIME = (9, 32)` controls protective TRAIL stop
     activation.
   - New pre-09:32 audit-created TRAIL orders set
     `goodAfterTime=YYYYMMDD 09:32:00 US/Eastern`.
   - New entry bracket parent BUY and child TRAIL orders share the same
     `goodAfterTime` value when submitted before 09:45 ET; after 09:45 ET the
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

   Live autotrader was restarted with `clientId=1` and reached the then-normal
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
     ... &` returned but did not keep the trader process alive reliably.
     `setsid -f ./scripts/start_trader.sh live >
     logs/live_autotrader_stdout.log 2> logs/live_autotrader_stderr.log <
     /dev/null` did keep it alive.
   - `scripts/start_trader.sh` now includes a lightweight supervisor loop when
     `VELOCITY_TRADER_AUTO_RESTART=1` (default). It restarts `auto_trader.py`
     after unexpected process exit and can be disabled by creating
     `${VELOCITY_BASE_DIR}/DISABLE_AUTO_RESTART`.
   - Live supervisor validation: killed child PID 1999277; parent
     `start_trader.sh live` stayed alive, logged exit status 143, waited 30
     seconds, and restarted child PID 2000214. The restarted child reconnected
     to IBKR and re-confirmed RIOT protective TRAIL order 7546.
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
   - Full suite: 348 passed after Python fixes and again after supervisor
     script changes.
   - `py_compile`: passed.

24. Startup and post-open broker-state audits, 2026-05-29

   - `PRE_ENTRY_SYNC_TIME` is `09:15 ET`. This is the early morning account,
     position, and stop-order health check.
   - Startup also runs one immediate position sync and protective-stop audit.
     It must not sleep until 09:15 ET, because the 06:30 ET full-universe
     prefilter runs from the main loop.
   - `POST_OPEN_AUDIT_TIME` is `09:35 ET`. This is a separate mandatory
     post-open position sync and protective-stop audit after the 09:30 opening
     auction and after TRAIL orders become active at 09:32 ET.
   - The 09:35 audit is independent from the early audit; it is not skipped
     simply because the 09:15 audit already ran.
   - If a normal protective stop audit already ran after 09:35 ET in the same
     trading day, that audit counts as satisfying the post-open checkpoint so
     the engine does not duplicate broker calls in the same cycle.

1. Live-session bug fixes, 2026-05-29

   Four bugs identified from live log analysis and fixed in commit `28753bf`.

   Stop-order race condition (cash account): entry used a bracket order
   pattern (`BUY transmit=False` + child `TRAIL SELL` with `parentId`). IBKR
   evaluates the child `SELL` before the parent `BUY` settles as a long
   position in a cash account and rejects it as a short, leaving new positions
   momentarily without a protective stop. Fix: replaced the bracket with a
   sequential pattern. The `BUY` is transmitted standalone (`transmit=True`);
   after `loopUntil()` confirms the fill, the engine places the `TRAIL SELL`
   as a completely independent `GTC` order with no `parentId`. This matches
   the proven standalone stop path used by `_audit_stop_orders()`. If the
   stop is rejected or the pre-flight fails, `_audit_stop_orders()` is called
   immediately.

   Correlation check blocking all candidates after a new entry:
   `_compute_book_correlation()` called
   `pd.concat([cand_ret, book_ret], axis=1, join='inner')` on two Series with
   independent integer row-indices from separate `reqHistoricalData` calls
   (e.g., index 0–349 for a `DAILY_LOOKBACK` fetch vs. index 0–59 for a `'90
   D'` on-demand fetch). The inner join on mismatched integer ranges produced
   zero rows, triggering the `< 20` threshold and failing closed (returning
   `1.0`), which silently blocked every candidate from entering against a
   recently-bought book symbol. Fix: added `_daily_returns()` static method
   that sets the `'date'` column as the DataFrame index before computing
   returns so the inner join aligns by calendar date. Freshly-fetched book
   bars are now also cached in `_bar_cache` so subsequent candidates in the
   same scan cycle reuse them.

   VIX live ticker always returning no price: `reqTickers()` for the VIX
   index with delayed data type 3 returns a ticker where `marketPrice()` and
   `close` are both empty. The `prevClose` and `last` fields, which delayed
   subscriptions do populate, were not checked. Fix: added `last` and
   `prevClose` field checks. Downgraded the ticker-miss log from `WARNING` to
   `INFO` since this is expected behaviour with delayed subscriptions. The
   historical fallback on error now returns the last cached VIX value instead
   of `None` so entries are not unnecessarily blocked.

   VIX timeout storm causing connection crashes: `_fetch_vix_price()` issued
   a `reqHistoricalData` call for VIX on every cycle (~60 s). When IBKR's
   HMDS feed was slow these calls piled up, each timing out and consuming a
   `reqId` slot, eventually triggering Error 1100 and a
   `ConnectionError: Socket disconnect` engine crash. Fix: added a 5-minute
   TTL cache (`_last_vix_ts`). `_fetch_vix_price()` returns the cached value
   immediately if it is fresher than 300 seconds. On historical-fallback
   failure, the stale cached value is returned rather than `None`.

   Pre-existing test fix: `test_run_cycle_skips_entries_when_account_summary_unavailable`
   was already failing before these changes. `_maybe_run_off_hours_jobs()`
   (added in the previous commit) calls `_write_dashboard_data()` during
   post-close maintenance, adding a second dashboard write when the test runs
   after market hours. Fix: patch `_maybe_run_off_hours_jobs` in the test and
   change `assert_called_once_with` to `assert_called_with`.

   Validation after these fixes:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py
   ```

   Results:

   - Full suite: 365 passed.
   - `py_compile`: passed.
   - Engine restarted live at 11:21:51 ET; both PLTR and HPQ TRAIL SELL stops
     confirmed on startup; no correlation errors or VIX warnings in the first
     scan cycle.

2. Off-hours maintenance validation follow-up, 2026-05-31

   The off-hours readiness/reconciliation code was revalidated after the
   environment date changed. A wall-clock-dependent test was corrected so it
   explicitly exercises the active regular-session management path instead of
   inheriting the actual weekend/off-hours clock.

   Added direct test coverage proving:

   - Premarket readiness writes `readiness_snapshot.json`, reconciles positions,
     audits protective stops, refreshes prices, and does not call velocity
     exits or scanner/new-entry logic.
   - Post-close reconciliation writes the same readiness snapshot without
     placing entries.
   - Off-hours maintenance checkpoints run only once per trading date.
   - Test helpers initialize `_last_vix_ts` and off-hours checkpoint dates so
     tests match production engine state.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity-pycache VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/config.py src/ib_gateway.py tests/test_engine.py
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q -p no:cacheprovider
   ```

   Results:

   - Full suite: 373 passed.
   - `py_compile`: passed.

3. HMDS/VIX reliability hardening, 2026-06-01

   Live session showed IBKR Historical Market Data Service instability:
   repeated VIX fallback timeouts, SPY historical timeouts, `2105` HMDS
   broken messages, and socket disconnects. Since SPY also timed out, this was
   diagnosed as a broader IBKR HMDS/session issue rather than simply missing
   VIX permissions.

   Fixes implemented:

   - Added SPY-then-VIX historical warmup after connect/reconnect. SPY failure
     now identifies general HMDS failure; SPY success + VIX failure identifies
     a VIX-specific data/entitlement problem.
   - Added VIX failure cooldown so repeated HMDS failures do not trigger
     historical requests every scan cycle. Fresh VIX cache is still allowed;
     stale VIX still fails closed and blocks new entries.
   - Added VIX data health fields to the health report: source, last success,
     failure count, last failure, next retry, and historical data probe status.
   - Added bounded historical request timeouts for warmup/fallback probes.
   - Reworked real IB account-summary requests to cancel the low-level
     `reqAccountSummary` subscription in `finally`, avoiding leaked summary
     subscriptions after disconnects/timeouts.
   - Treated socket disconnects during sleep as reconnectable events instead
     of logging them as generic runtime crashes.

   Validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity_full_tests PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   .venv/bin/python run_backtest.py --yearly --start 2020-01-01
   ```

   Results:

   - Full suite: 384 passed.
   - Yearly full-universe backtest completed from 2020-01-01 through
     2026-05-01 using cached data; operational changes did not alter strategy
     rules.

4. Shared scoring experiment, 2026-06-02

   Added `src/scoring.py` so live trading and backtesting can use the same
   candidate ranking functions. The live scanner now stores both raw intraday
   RVOL and time-normalized volume pace; live ranking uses volume pace so early
   regular-session candidates are not punished simply because only part of the
   day has elapsed.

   Added optional `VELOCITY_SCORING_MODEL=enhanced` / `--scoring-model enhanced`
   research mode with:

   - Softer RSI-over-75 treatment.
   - Extension quality: rewards clean ORB breakouts and penalizes stretched
     chase entries.
   - ATR risk quality: prefers cleaner movers over wild ATR% names.
   - Shared live/backtest scoring path.

   Promotion decision:

   - Do not promote enhanced scoring as production default.
   - Keep `VELOCITY_SCORING_MODEL=legacy` for paper/live trading.
   - Reason: enhanced was slightly better on the cached 282-symbol validation
     set, but materially worse on the full cached 3,058-symbol universe.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m py_compile src/scoring.py src/engine.py src/config.py backtest/strategy.py backtest/optimizer.py run_backtest.py
   VELOCITY_BASE_DIR=/tmp/velocity_full_tests PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   PYTHONUNBUFFERED=1 VELOCITY_BASE_DIR=/tmp/velocity_bt_full_legacy PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 0 --scoring-model legacy
   PYTHONUNBUFFERED=1 VELOCITY_BASE_DIR=/tmp/velocity_bt_full_enhanced PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 0 --scoring-model enhanced
   ```

   Results:

   - Full suite: 390 passed.
   - Full universe legacy: 10,459 trades, +1,314,669.32%, Win 78.9%, PF 11.47,
     MaxDD -5.11%, Sharpe 8.42.
   - Full universe enhanced: 10,896 trades, +508,686.95%, Win 78.5%, PF 8.48,
     MaxDD -5.38%, Sharpe 8.41.
   - Bounded 282-symbol validation favored enhanced slightly on aggregate, but
     full-universe validation overrules it for production.

5. Legacy v2 scoring promotion, 2026-06-02

   Implemented a smaller, safer improvement to legacy scoring instead of
   replacing the model with the broader enhanced scorer. `legacy_v2` keeps the
   original legacy score dominant and adds bounded tie-breakers for:

   - Volume pace follow-through.
   - Dollar-liquidity depth.
   - Clean ORB extension without chasing stretched names.
   - ATR risk cleanliness.
   - Mild high-RSI relief only when RSI is still rising.

   Promotion decision:

   - Promote `legacy_v2` as the default scoring model.
   - `legacy` remains available for comparison and rollback.
   - `enhanced` remains research-only and should not be used as production
     default based on the full-universe result above.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m py_compile src/scoring.py src/engine.py src/config.py backtest/strategy.py backtest/optimizer.py run_backtest.py
   VELOCITY_BASE_DIR=/tmp/velocity_full_tests PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   PYTHONUNBUFFERED=1 VELOCITY_BASE_DIR=/tmp/velocity_bt_full_legacy_v2 PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 0 --scoring-model legacy_v2
   ```

   Results:

   - Full suite: 394 passed.
   - Full universe legacy baseline from the same cached 3,058-symbol universe:
     10,459 trades, +1,314,669.32%, Win 78.9%, PF 11.47, MaxDD -5.11%,
     Sharpe 8.42.
   - Full universe legacy_v2: 10,623 trades, +1,320,010.65%, Win 79.0%,
     PF 10.76, MaxDD -4.93%, Sharpe 8.46.
   - Legacy v2 improved return, drawdown, Sharpe, win rate, and trade count.
     Profit factor declined but remained very high; net risk-adjusted result
     justified promotion.

6. Live/backtest entry-rule sync, 2026-06-02

   Reviewed the live entry path against the daily-bar backtester after the
   `legacy_v2` scoring promotion. The core 8096 gates were aligned except for
   one important mismatch:

   - Backtest gap cap checked the signal day's open against `prev_high`.
   - Live gap cap checked the current live price against ORB.

   Fix:

   - Live scan gate now uses `day_open <= orb_high * (1 + active_gap_cap)`.
   - Live pre-order reprice validation uses the same opening-gap definition.
   - Current-price extension is left to scoring/ranking quality, matching the
     backtester where the completed close can extend beyond the opening-gap cap.

   Live-only execution safeguards intentionally remain live-only:

   - Bid/ask spread gate.
   - Correlation and sector concentration gates.
   - Friday entry cutoff and Friday liquidity multiplier.
   - Broker/order preflight and spread-aware limit pricing.

   These are not removed because daily-bar backtests cannot model them
   reliably, and they are production safety controls rather than alpha-entry
   rule drift.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m py_compile src/engine.py src/config.py src/scoring.py backtest/strategy.py tests/test_engine.py tests/test_backtest.py tests/test_trailing_stop_scoring_screener.py
   VELOCITY_BASE_DIR=/tmp/velocity_full_tests PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   ```

   Results:

   - Focused engine/backtest/scoring tests: passed.
   - Full suite: 397 passed.

7. Live exit-policy cleanup, 2026-06-02

   Reviewed the June 1 live logs after OXY was closed at 15:46 ET. The close
   was not a velocity exit; it was the live `EOD FLAT` rule liquidating a
   same-day position that was down about 1%. That behavior was too aggressive
   for a swing system and was not aligned with the stated minimum hold policy.

   Fix:

   - Renamed the live exit manager to `manage_position_exits()`.
   - Updated engine call sites and tests to use `manage_position_exits()`.
   - Removed the stale `check_velocity_exits()` compatibility wrapper during the 2026-06-04 cleanup pass.
   - Historical note: this pass temporarily stopped EOD flat from closing
     same-day swing entries. Sections 8 and 10 supersede that behavior; current
     EOD quality cleanup is same-day.
   - Software exits now require a fresh broker price; cached `current_price`
     and stale ticker `close` values are no longer allowed to liquidate a
     position.
   - `_fresh_market_price()` no longer uses `ticker.close` as an exit price
     source. It uses market price, last, bid/ask midpoint, or bid.

   Exit policy after the fix:

   - Broker trailing stop: primary always-on protection.
   - Hard stop: live software backup during the regular management window, with
     fresh broker price only.
   - Break-even giveback: fresh broker price only.
   - Friday close: explicit weekend-risk policy.
   - EOD quality cleanup: same-day at/after 15:50 ET, using the quality-based
     hold gate documented in section 10.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_exit_tests .venv/bin/python -m pytest -q tests/test_engine.py::TestVelocityExit tests/test_trailing_stop_scoring_screener.py::TestExitOrders tests/test_trailing_stop_scoring_screener.py::TestHardStop tests/test_trailing_stop_scoring_screener.py::TestFridayClose tests/test_trailing_stop_scoring_screener.py::TestEodFlat -p no:cacheprovider
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m py_compile src/engine.py src/config.py tests/test_engine.py tests/test_trailing_stop_scoring_screener.py
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_full_tests .venv/bin/python -m pytest -q -p no:cacheprovider
   PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_bt_exit_policy .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 0 --scoring-model legacy_v2
   ```

   Results:

   - Focused exit-policy tests: 32 passed.
   - Full suite: 398 passed.
   - Full-universe backtest: 10,623 trades, +1,320,010.65%, Win 79.0%,
     PF 10.76, MaxDD -4.93%, Sharpe 8.46.
   - Backtest metrics stayed aligned with the promoted `legacy_v2` production
     baseline because this was a live execution-policy fix, not a backtest alpha
     rule change.

8. EOD profit cleanup consolidation, 2026-06-02

   Removed the separate live velocity-exit rule and folded the stale-capital
   logic into one end-of-day profit cleanup rule.

   Fix:

   - Removed `VELOCITY_EXIT_TIME` from `src/config.py`.
   - Moved `EOD_EXIT_TIME` from `15:45 ET` to `15:50 ET`.
   - Live `manage_position_exits()` now has no separate velocity liquidation
     branch.
   - EOD cleanup rule is same-day: if profit is below `PROFIT_MIN_THRESHOLD`
     at/after `15:50 ET`, close via market sell to free capital for T+1
     settlement.
   - Hard stop, break-even giveback, Friday close, and broker trailing stops
     remain independent safety exits and are not delayed to 15:50 ET.
   - Dashboard rule text now says:
     `Same day at/after 3:50 PM ET: if profit < 5%, force-liquidate via Market SELL; frees capital for T+1 settlement`.
   - `/api/state` now exposes `eod_exit_time`.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_same_day_eod_tests .venv/bin/python -m pytest -q tests/test_engine.py::TestEodProfitCleanup tests/test_trailing_stop_scoring_screener.py::TestExitOrders tests/test_trailing_stop_scoring_screener.py::TestEodFlat tests/test_dashboard_server.py tests/test_backtest.py::TestOptimizerHelpers -p no:cacheprovider
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache .venv/bin/python -m py_compile src/engine.py src/config.py dashboard_server.py backtest/strategy.py backtest/optimizer.py run_backtest.py tests/test_engine.py tests/test_trailing_stop_scoring_screener.py tests/test_dashboard_server.py tests/test_backtest.py
   PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_full_tests .venv/bin/python -m pytest -q -p no:cacheprovider
   PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/velocity_pycache VELOCITY_BASE_DIR=/tmp/velocity_bt_same_day_eod .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --scoring-model legacy_v2
   ```

   Results:

   - Focused same-day EOD cleanup/dashboard/optimizer tests: 38 passed.
   - Syntax compile: passed.
   - Full suite: 401 passed.
   - Cached 282-symbol forward backtest, 2020-01-01 through 2026-05-22:
     3,018 trades, +2,939.50% total return, 92.1% win rate, 119.24 profit
     factor, -4.58% max drawdown, 6.27 Sharpe.
   - Backtest exit breakdown now reports `eod_profit_cleanup` instead of
     `velocity_exit`.
   - Important research caveat: same-day daily-bar cleanup fills weak entries at
     the completed daily close. This matches the intended live 15:50 ET rule at
     a coarse level, but it is still less realistic than an intraday replay.

9. Plug-and-play strategy profiles, 2026-06-06

   Screening and entry rules are now selectable profiles. Exit rules remain
   unchanged.

   Profiles:

   - `current`: existing ORB momentum/risk filter.
   - `reversal_reclaim`: lower-priced reclaim/reversal momentum sleeve.
   - `five_day_momentum`: 5-day momentum sleeve near 20-day highs.
   - `safer_liquid_momentum`: more liquid, tighter-risk momentum sleeve.

   Implementation:

   - Added `src/strategy_profiles.py` as the single source of truth for profile
     scan codes, scanner-side generic filters, thresholds, and entry checks.
   - Live scanner uses profile scan codes unless
     `VELOCITY_IB_SCANNER_SCAN_CODES` explicitly overrides them.
   - Live entry and backtest fine-entry logic now call the same shared evaluator.
   - Backtest CLI now supports
     `--strategy-profile {current,reversal_reclaim,five_day_momentum,safer_liquid_momentum}`
     plus `--min-price`, `--min-volume`, and `--min-dollar-vol` overrides.
   - Backtest cache keys now include profile and universe-floor values so
     strategy sleeves cannot silently reuse the wrong cached universe.

   Validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m py_compile src/strategy_profiles.py src/scanner.py src/engine.py backtest/strategy.py backtest/optimizer.py run_backtest.py
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -o cache_dir=/tmp/velocity-pytest-cache -q
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python run_backtest.py --help
   ```

   Results:

   - Syntax compile: passed.
   - Full suite: 409 passed.
   - Backtest CLI exposes all four strategy profiles.

10. EOD quality-based hold rule, 2026-06-06

   The old same-day EOD cleanup was a blunt `profit < 5%` liquidation rule.
   That has been replaced with a quality gate for deciding whether a position
   deserves overnight capital.

   New live EOD rule at/after `15:50 ET`:

   - Hold only if profit is at least `EOD_HOLD_MIN_PROFIT_PCT` (default `0%`).
   - Price must be above VWAP when VWAP is available, or strictly above entry.
   - Price must be near the day high:
     `day_range_location >= EOD_HOLD_DAY_RANGE_LOCATION_MIN` (default `0.70`).
   - Intraday relative strength versus SPY must be positive:
     stock intraday return minus SPY intraday return >=
     `EOD_HOLD_RELATIVE_STRENGTH_MIN` (default `0%`).
   - Protective stop must be confirmed when
     `EOD_HOLD_REQUIRE_STOP_CONFIRMED=1` (default).

   Fail-closed behavior:

   - If fresh price, day range, stock open, SPY return, or confirmed stop data is
     unavailable, the EOD hold test fails and the engine liquidates the position.
   - Exit snapshots intentionally do not use stale `ticker.close` as a price
     fallback.

   Backtest approximation:

   - Daily backtests use close >= entry as the no-look-ahead proxy for the
     live VWAP/entry condition.
   - Daily close location uses `(close - low) / (high - low)`.
   - Daily relative strength uses stock daily return minus SPY daily return.
   - The legacy `PROFIT_MIN_THRESHOLD` parameter remains only for optimizer
     compatibility.

   Validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m py_compile src/config.py src/engine.py backtest/strategy.py backtest/optimizer.py dashboard_server.py run_backtest.py tests/test_engine.py tests/test_trailing_stop_scoring_screener.py tests/test_backtest.py tests/test_dashboard_server.py
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -o cache_dir=/tmp/velocity-pytest-cache tests/test_engine.py::TestEodProfitCleanup tests/test_trailing_stop_scoring_screener.py::TestExitOrders tests/test_trailing_stop_scoring_screener.py::TestFridayClose tests/test_trailing_stop_scoring_screener.py::TestEodFlat tests/test_dashboard_server.py tests/test_backtest.py -q
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -o cache_dir=/tmp/velocity-pytest-cache -q
   ```

   Results:

   - Syntax compile: passed.
   - Focused EOD/dashboard/backtest tests: 97 passed.
   - Full suite: 410 passed.

11. Trader stale-heartbeat watchdog, 2026-06-08

   Live issue observed:

   - The dashboard server stayed alive and `/api/state` was reachable, but the
     trader child stopped writing fresh `runtime/live/dashboard_data.json` and
     `runtime/live/engine_state.json`.
   - The Python `auto_trader.py` process was still present, so the existing
     auto-restart loop did not help because it only restarted after process
     exit.

   Fix:

   - `scripts/start_trader.sh` now has a supervisor-level heartbeat watchdog.
   - The watchdog watches `${VELOCITY_BASE_DIR}/dashboard_data.json` by default.
   - If the heartbeat file is stale beyond
     `VELOCITY_TRADER_STALE_SEC` after
     `VELOCITY_TRADER_WATCHDOG_STARTUP_GRACE_SEC`, the supervisor terminates
     the stuck child and the normal restart loop starts a fresh trader.
   - Defaults are conservative to avoid killing a slow but legitimate scanner
     cycle:
     `VELOCITY_TRADER_STALE_SEC=600`,
     `VELOCITY_TRADER_WATCHDOG_INTERVAL_SEC=15`,
     `VELOCITY_TRADER_WATCHDOG_STARTUP_GRACE_SEC=900`.
   - `.env.paper.example` and `.env.live.example` document the knobs.

   Validation commands:

   ```bash
   bash -n scripts/start_trader.sh
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -o cache_dir=/tmp/velocity-pytest-cache tests/test_start_trader_supervisor.py -q
   ```

1. Tiered exits removed and Chandelier multiplier changed to 1×, 2026-06-20

   Decision: remove tiered profit exits and break-even exit; evaluate 1× ATR
   trailing stop for ~2 weeks of live trading before deciding whether to adjust.

   Changes:
   - Removed `TIERED_PROFIT_EXIT_ENABLED`, `TIERED_PROFIT_EXIT_R_LEVELS`,
     `BREAK_EVEN_R_MULT`, `BREAK_EVEN_PEAK_RETAIN_FRACTION`, and
     `MIN_ENTRY_SHARES` from `src/config.py`.
   - Removed all tiered/break-even methods from `src/engine_orders.py`,
     `src/engine_exits.py`, `src/engine_entries.py`, `src/engine.py`,
     `backtest/strategy.py`, and `backtest/optimizer.py`.
   - Changed `CHANDELIER_MULT` from `2.0` to `1.0` in `src/config.py`.
   - Optimizer grid updated to `[0.8, 1.0, 1.2]` (default) and `[0.8, 1.0]`
     (quick) to test around the new default.
   - Dashboard updated: removed PROFIT TIERS column, fixed `colspan` from 17
     to 16, updated entry/exit condition descriptions to show actual config
     thresholds, made Chandelier multiplier dynamic via token injection.
   - Renamed `CODEX.md` → `CLAUDE.md`.

   Always start the application using the profile scripts, not bare Python:

   ```bash
   # live trading (port 4001)
   nohup ./scripts/start_trader.sh live > logs/live_autotrader_stdout.log 2> logs/live_autotrader_stderr.log < /dev/null &
   nohup ./scripts/start_dashboard.sh live > logs/live_dashboard_stdout.log 2> logs/live_dashboard_stderr.log < /dev/null &

   # paper trading (port 4002)
   nohup ./scripts/start_trader.sh paper > logs/paper_autotrader_stdout.log 2> logs/paper_autotrader_stderr.log < /dev/null &
   nohup ./scripts/start_dashboard.sh paper > logs/paper_dashboard_stdout.log 2> logs/paper_dashboard_stderr.log < /dev/null &
   ```

   Running `nohup .venv/bin/python auto_trader.py` without env vars defaults
   to paper mode (port 4002) and uses the project root as `VELOCITY_BASE_DIR`
   instead of `runtime/live` or `runtime/paper`.

   Validation:

   ```bash
   VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -q --tb=short
   ```

   Results: 342 passed.

1. Replaced ATR trailing stop with flat percent trailing stop, 2026-06-24

   Decision: remove ATR/Chandelier-based stop; use a simple flat 4% percent trailing
   stop (`TRAIL_PCT = 0.04`) that is price-proportional and requires no ATR calculation
   at entry time. IBKR TRAIL order uses `trailingPercent` instead of `auxPrice`.

   Changes:
   - Removed `CHANDELIER_MULT` from `src/config.py`; added `TRAIL_PCT = 0.04`.
   - `CHANDELIER_PERIOD` retained for entry volatility filter (ATR_CHAND/price gate),
     not for stop distance.
   - `src/engine.py`: `trail_dist = round(limit_price * TRAIL_PCT, 2)`, TRAIL order
     uses `trailingPercent` instead of `auxPrice`. State includes `stop_mode='percent'`
     and `trailing_percent=4.0`.
   - `src/engine_orders.py`: audit no longer fetches historical bars; computes
     `trail_dist` from `entry_px * TRAIL_PCT` and places with `trailingPercent`.
     `_trail_order_protection` updated to handle `trailingPercent`-only orders
     (where `trailStopPrice` is UNSET until broker populates it).
   - `src/engine_entries.py`: `effective_stop` display uses `peak × (1 - TRAIL_PCT)`
     instead of static initial_sl.
   - `backtest/strategy.py`: stop = `peak_high × (1 - trail_pct)`. Position sizing
     uses `entry_price × trail_pct` for stop distance.
   - `backtest/optimizer.py`: `chandelier_mult` → `trail_pct`, grid `[0.03, 0.04, 0.05]`.
   - `run_backtest.py`: `--chandelier-mult` → `--trail-pct`.
   - `dashboard_server.py`: "Percent Trail" row with `__TRAIL_PCT__%` token.
   - All tests updated; 164 passed.

1. In-the-money trailing-stop reset bug fixed, 2026-07-01

   Live-log analysis of the AMD trade (bought 2026-06-30 at $565.24, 1 share,
   2% percent TRAIL) found a real defect that turned a locked winner into a
   loss. AMD ran up and the broker-side percent TRAIL trailed its
   `trailStopPrice` up to $573.00. On 2026-07-01 pre-market the stock pulled
   back to ~$564.88 — below the trailed stop. The 08:45 ET stop audit saw
   `trail stop $573.00 >= reference price $564.88`, wrongly classified the
   order as invalid, cancelled it, and rebuilt a fresh 2% trail from the
   fallen price ($547.43). AMD then fell through the reset stop and exited
   near ~$547 instead of the locked ~$573.

   Root cause: `OrdersMixin._trail_order_protection()` in `src/engine_orders.py`
   rejected any percent trail whose `trailStopPrice >= ref_price`. Because
   `_coerce_order_number()` already maps IBKR `UNSET_DOUBLE` to `None`, a
   non-`None` `trail_stop` in that branch is always a real, finite stop price,
   so the check only ever fired on legitimate in-the-money trailed stops. A
   SELL stop at/above the current price is a *breached* stop that should
   trigger — not a malformed order.

   Fix:

   - Removed the `trail_stop >= ref_price` rejection in the percent-trail
     branch. A finite `trailStopPrice` with a sane `trailingPercent` (< 99) is
     now always treated as valid protection.
   - Stop distance is computed safely: `ref_price - trail_stop` when the ref
     price is above the stop, otherwise the trail-percent-implied distance
     `trail_stop * (trail_pct / (100 - trail_pct))`, so a non-positive gap
     never leaks downstream.
   - Left untouched (correct guards): the dollar-trail `aux_dist >= ref_price`
     check, the "missing dollar/percent fields" rejection, GTC persistence, and
     the 09:32 ET RTH-only stop activation gate.

   Regression test added:
   `tests/test_startup_init.py::TestAuditStopOrders::test_keeps_in_the_money_percent_trail_above_reference_price`
   — an existing percent TRAIL with `trailStopPrice=$573` above the current
   price ($564.88) must be kept (no cancel, no rebuild) with
   `stop_loss`/`effective_stop` = $573.

   Also fixed a stale, unrelated test left behind by commit `8588cbf` (which
   removed the dashboard `R` and `RS 63D` columns):
   `tests/test_dashboard_server.py::test_dashboard_equity_chart_uses_intraday_time_labels`
   asserted `<th>R</th>` still existed. It now enforces that both removed
   columns stay gone.

   Validation:

   ```bash
   PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/engine_orders.py src/config.py src/ib_gateway.py
   PYTHONPYCACHEPREFIX=/tmp/velocity-pycache VELOCITY_BASE_DIR=/tmp/velocity-test .venv/bin/python -m pytest -o cache_dir=/tmp/velocity-pytest-cache -q
   ```

   Results:

   - `py_compile`: passed.
   - Full suite: 345 passed (344 prior passing + the new regression test; the
     previously stale dashboard test now passes).
   - Strategy rules unchanged — this is a broker-state/audit correctness fix,
     not an alpha or exit-policy change.

1. Brutal live-performance review and three live-risk fixes, 2026-07-22

   The trade ledger (built 2026-07-11-ish, see the ledger sections above) gave
   the first real measurement of live performance: 11 closed trades since
   deployment, 27.3% win rate, 0.16 profit factor, net -$78.44, account equity
   -12.1% from its first recorded peak ($2082.27 on 2026-05-28 to $1830.36 on
   2026-07-22). Median MFE on the 10 `trail_stop` exits was 2.048% — almost
   exactly the live 2% trail width, i.e. real trades were running up to the
   stop distance and reversing before any real move developed. This is the
   live-money confirmation of the 2026-07-10 walk-forward optimizer result
   that was written down but never deployed. Three fixes were authorized and
   applied from that review:

   - **`TRAIL_PCT` promoted from 0.02 to 0.05** (`src/config.py`). This is the
     already-validated 2026-07-10 walk-forward optimum, applied 12 days after
     it was first recommended. Promoted as the code *default* (not a
     live-only env override) since it is a validated strategy parameter, same
     pattern as the `RECLAIM_TRIGGER_BONUS` promotion — it now also applies to
     paper trading and the backtester baseline.
   - **Live cash buffer widened**: `VELOCITY_SETTLED_CASH_DEPLOYMENT_PCT=0.85`
     added to `.env.live.local` (live-only override; code default stays 0.95
     for paper/backtest). The code default only ever reserved ~5% of settled
     cash, which on this account's 3-slot sizing left $16-55 free for days at
     a time with 0 entry slots. 0.85 reserves ~15% instead. This did not
     require a code change — `SETTLED_CASH_DEPLOYMENT_PCT` already existed
     and was already wired into both `EntriesMixin._deployable_settled_cash()`
     and the backtester's equivalent; only the live value needed widening.
   - **New periodic momentum-stall exit** (`src/engine_exits.py`,
     `src/config.py`): added at explicit user request, **not yet
     backtest-validated**. Every `MOMENTUM_HOLD_CHECK_INTERVAL_DAYS` (default
     3 calendar days), a held position must show at least
     `MOMENTUM_HOLD_MIN_MOVE_PCT` (default +1%) close-to-close appreciation
     over the trailing `MOMENTUM_HOLD_LOOKBACK_DAYS` (default 2) completed
     daily sessions, checked via `ExitsMixin._momentum_hold_passes()`, or the
     position is closed via `liquidate(sym, reason='momentum_stall')` to
     recycle capital. This is a momentum-*continuation* check, independent of
     profit/loss versus entry — it can close a position that is still up
     overall if it has stopped making fresh progress. Uses completed daily
     closes (via `completed_daily_bars()`), not the live intraday price, so
     the judgement isn't made on an in-progress bar. Cadence is tracked per
     symbol via a new `momentum_check_date` state field; the check only runs
     once `trading_bars_held >= MOMENTUM_HOLD_LOOKBACK_DAYS`. On a daily-
     history fetch failure it fails open (retries next cycle, does not stamp
     `momentum_check_date`) rather than liquidating on a data outage.
     Controlled by `VELOCITY_MOMENTUM_HOLD_ENABLED` (default on). Because this
     rule has no backtest coverage yet, it should be watched closely in live
     logs (`MOMENTUM STALL EXIT` / `MOMENTUM HOLD` lines) and is a candidate
     for a proper backtest implementation + forward validation before being
     trusted the way `TRAIL_PCT` or the reclaim bonus are.

   Also updated: `tests/test_trailing_stop_scoring_screener.py` —
   `TestBracketOrderMath`'s `TRAIL_DIST`/`INIT_STOP` constants and the
   "TRAIL_PCT is two percent" test were hardcoded to the old 0.02 default;
   made dynamic against `src.config.TRAIL_PCT` (module-level `_TRAIL_PCT`
   import) so the next promotion doesn't require another test edit, and the
   percent-value assertion now pins 0.05 with the promotion rationale in the
   docstring. Added `TestMomentumStallExit` (6 tests): stall exit fires below
   threshold, hold passes at/above threshold, check skipped before
   `MOMENTUM_HOLD_LOOKBACK_DAYS` held, check not repeated inside the cadence
   interval, fails open on daily-history fetch failure, and the config kill
   switch fully disables the rule.

   Validation:

   ```bash
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/engine_exits.py src/engine_entries.py src/engine_orders.py src/engine_market.py src/engine_scanner.py src/engine_base.py src/config.py src/strategy_profiles.py backtest/strategy.py backtest/optimizer.py run_backtest.py tests/test_trailing_stop_scoring_screener.py
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   ```

   Results:

   - `py_compile`: passed.
   - Full suite: 379 passed.
   - Live trader and dashboard restarted under the new config; live logs
     confirmed `trail_pct=5.0%` on the next order and `SETTLED_CASH_DEPLOYMENT_PCT`
     effective at 0.85 (see the restart record immediately below, if present).

1. Daily stop-gate refresh was silently resetting ratcheted trailing stops, 2026-07-27

   A full application review (requested after stopping the live stack) found
   that the two trades closed since the `TRAIL_PCT=5%` promotion (ADM, ARWR)
   both exited via `momentum_stall`, never `trail_stop` — even though ARWR had
   genuinely run up to a **$92.675 peak (+6.6% MFE)** on 2026-07-23/24. A 5%
   trail from that peak should have protected a stop near $88.04. Live logs
   showed the actual sequence:

   ```
   07-27 00:15 EDT | AUDIT: ARWR — existing TRAIL SELL has stale/missing stop
                     activation gate; cancelling and replacing with
                     goodAfterTime=20260727 09:32:00 US/Eastern
   07-27 00:15 EDT | AUDIT: ARWR — TRAIL SELL confirmed (id=182350 stop=$84.69
                     trail=5%)
   ```

   $84.69 / 0.95 = an implied reference of **$89.15 — a $3.53 (3.8%) gap from
   the true $92.675 peak**, not explainable by rounding (the companion ADM
   trade, which never ran further than its own entry, showed only a 4-cent
   gap — confirming the math and showing the effect only bites once a
   position has genuinely run up). ARWR then drifted down and was closed by
   the (separately unvalidated) momentum-stall rule at $85.395 — a loss on a
   trade that had been a real +6.6% winner four days earlier.

   Root cause: `_audit_stop_orders()` runs `_replace_trail_with_stop_activation_gate()`
   on every valid `trail_orders[0]` whenever the order's `goodAfterTime`
   string doesn't match today's freshly computed stop gate. Because the gate
   string is date-stamped (`"YYYYMMDD 09:32:00 US/Eastern"`), it goes stale
   **every single day** an order is held overnight — so this fired on
   essentially the first audit of every new day for every multi-day position,
   not as a rare edge case. Each firing does a genuine IBKR cancel + place of
   a brand-new order (new orderId). The replacement code tried to carry
   forward the old order's `trailStopPrice`/`trailingPercent`, but a fresh
   IBKR order has no memory of price history before its own creation —
   whatever mechanism is responsible (IBKR re-anchoring the trail's
   high-water mark to the price prevailing at creation time, or our own read
   of the old order's `trailStopPrice` not reflecting the true current
   ratcheted level), the net effect is the same: **the "stop ratchets up and
   never gives it back" guarantee was being silently violated once a day for
   every position held more than one session.** This is a different code path
   than the 2026-07-01 AMD fix (which stopped the audit from *misclassifying*
   a healthy in-the-money trail as invalid via `_trail_order_protection()`);
   this bug lives in the separate gate-refresh path that still touches orders
   already correctly classified as valid.

   Fix: `_replace_trail_with_stop_activation_gate()` now skips entirely
   (returns the order untouched) whenever the order's `orderId` matches
   `state[sym]['stop_order_id']` **and** `state[sym]['protection_status'] ==
   'confirmed'` — i.e. whenever we have already confirmed this exact order
   live in a prior audit cycle. A stale/mismatched `goodAfterTime` on an
   order we already know is active protects nothing further; the gate only
   has a real job to do for orders we have never confirmed (fresh entries,
   freshly rebuilt stops after a genuine repair, or an externally placed
   order the audit is seeing for the first time). This is a narrower, more
   precise fix than trying to parse/compare `goodAfterTime` timestamps
   directly, and it does not change behavior for any order that hasn't
   already been through a successful confirmation — the existing
   "gate a newly-discovered ungated order before the stop-activation time"
   safety net (`test_existing_trail_before_stop_gate_is_delayed_to_932_et`)
   is untouched because a brand-new/never-confirmed position's state has no
   matching `stop_order_id` yet.

   Regression tests added to `tests/test_startup_init.py::TestAuditStopOrders`:
   - `test_already_confirmed_trail_is_not_reset_for_stale_stop_gate` —
     reproduces the ARWR scenario exactly (confirmed order, stale dated gate,
     audit running before today's stop gate); asserts no cancel/replace and
     that `stop_loss`/`effective_stop` stay at the true ratcheted value.
   - `test_confirmed_status_for_different_order_id_still_gets_gate_replacement` —
     guards against an overly broad fix: a `protection_status='confirmed'`
     left over from a *different*, now-stale `stop_order_id` must not
     suppress gating for a genuinely new/different TRAIL order.

   Not yet done, deliberately: the live stack is currently stopped (no open
   positions) per an explicit prior instruction, so this fix has only been
   verified via unit tests reproducing the exact logged scenario, not by
   observing a real multi-day IBKR order survive a gate-refresh audit live.
   It should be watched closely (`AUDIT: ... TRAIL SELL confirmed` lines with
   no accompanying cancel/replace for known-good multi-day positions) the
   next time the stack runs live across a multi-day hold.

   Validation:

   ```bash
   .venv/bin/python -m py_compile auto_trader.py dashboard_server.py src/engine.py src/engine_orders.py src/config.py src/ib_gateway.py tests/test_startup_init.py
   VELOCITY_BASE_DIR=/tmp/velocity-test PYTHONPYCACHEPREFIX=/tmp/velocity-pycache .venv/bin/python -m pytest -q -p no:cacheprovider
   ```

   Results:

   - `py_compile`: passed.
   - Focused audit tests: 23 passed (21 prior + 2 new regression tests).
   - Full suite: 381 passed.
