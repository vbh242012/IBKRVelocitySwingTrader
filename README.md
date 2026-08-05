# IBKRVelocitySwingTrader
Automated Swing Trading For Small Cash Account With T+1 Settlement Days Using Interactive Broker

## Strategy

Two strategies can hold live positions, both gated by independent kill
switches in the profile's `.env.*.local` file. Both share the same broker
order-placement/protection machinery, the same slot pool, and the same
account-level safety controls (VIX regime filter, spread cap, correlation
cap, sector cap, settled-cash sizing, T+1 settlement).

### `indicator_swing` — relative-strength-first swing momentum

The primary, trend-following profile (`VELOCITY_INDICATOR_SWING_ENTRIES_ENABLED`,
default on):

- Relative-strength first. A stock must pass trend, liquidity,
  relative-strength, weekly-uptrend, MA50/MA200, 52-week-high proximity, ATR,
  spread, and volume gates before an indicator sleeve can buy.
- EMA20 above SMA50 is the default trend state; MA reclaims or prior-high
  breaks time entries after the RS gate passes. Bollinger lower-band reclaim
  and PSAR flip are optional additional sleeves inside this same
  trend/RS-gated profile (`VELOCITY_INDICATOR_SWING_STRATEGIES`), not enabled
  by default.
- RSI momentum/recovery plus at least two of MACD, stochastic, OBV, and PSAR
  must confirm. Volume pace is a separate hard gate, not a confirmation vote.
  Analyst ratings can adjust score but cannot create a buy by themselves. The
  default minimum entry score is 50.
- Each position stores the sleeve that opened it and exits on that sleeve's
  rule.
- Exits: a broker-side flat percent TRAIL order (`TRAIL_PCT`, default 5% from
  peak price) is the primary, always-on protection. A software hard stop and
  a same-day (15:50 ET) EOD quality-hold rule — liquidate unless price is
  above VWAP/entry, near the day high, and outperforming SPY intraday — back
  it up. A periodic momentum-stall check can also close a position that has
  stopped making fresh progress even while still profitable. Friday close and
  analyst-downgrade-with-price-confirmation are additional safety exits.

### Standalone Bollinger mean reversion

A separate, additive strategy (`VELOCITY_BOLLINGER_STANDALONE_ENABLED`,
default off) with no trend/RS gates at all — entry is liquidity + spread +
`BB_RECLAIM_LOWER` (two prior closes below the lower Bollinger band, then a
reclaim). Exit is midline reclaim, a tighter 5% hard stop, or a 7-day time
stop, on top of the same broker percent TRAIL every position gets. Capped to
`BOLLINGER_STANDALONE_MAX_OPEN` (default 1) concurrent position, drawn from
the same slot pool as `indicator_swing` rather than an additive extra slot.
See `src/bollinger_standalone.py`.

Check `.env.live.local` / `.env.paper.local` for which of the two is actually
enabled for new entries in a given deployment — both can be toggled
independently.

Analyst ratings are optional confirmation. Live and paper trading use a dated
local CSV first, Finnhub recommendation trends when `VELOCITY_FINNHUB_API_KEY`
is configured, and Yahoo/yfinance as the default free fallback via
`VELOCITY_ANALYST_RATINGS_FREE_SOURCE=yahoo`. Backtests do not fetch current
analyst ratings; they only use a dated local CSV snapshot from
`VELOCITY_ANALYST_RATINGS_FILE` so historical research does not cheat with
future information.

## Application Scanner

The live/paper application scanner can now use `VELOCITY_APP_SCANNER_SOURCE`:

- `ibkr`: only IBKR scanner subscription results.
- `universe`: a rotating batch from the full US common-stock universe.
- `hybrid`: IBKR scanner results plus the rotating universe batch.

`hybrid` is the default. The universe is loaded from
`VELOCITY_APP_SCANNER_UNIVERSE_FILE` when provided, otherwise from cached NASDAQ
Trader listing files. `VELOCITY_APP_SCANNER_BATCH_SIZE` controls how many
universe symbols are added per scan cycle. Final buy decisions still come from
the same local screener/profile rules inside the live engine.

At `VELOCITY_APP_PREFILTER_START_TIME` (`06:30 ET` by default), the app runs a
premarket historical universe sieve. It scans the full configured universe once,
rejects symbols that cannot satisfy static daily rules during the session
(history length, daily liquidity, MA structure, SMA200 slope, stable daily
momentum confirmations, and stable sleeve feasibility), and writes the surviving
candidate list to `premarket_universe_prefilter.json` in the runtime folder.
Startup enters the main loop immediately after the initial safety audit; the
09:15 ET pre-entry stop audit is a scheduled checkpoint, not a blocking startup
sleep, so the 06:30 prefilter can run on time.
During the entry window, the scanner uses that candidate list and only checks
rules that can still change intraday, such as live price, spread, volume pace,
reclaims, prior-high breaks, and orderability.
If IBKR historical pacing has not finished the full universe by `ENTRY_START`,
`VELOCITY_APP_PREFILTER_STOP_AT_ENTRY_START=1` saves a partial cache with
`stopped_reason=entry_window_open` and the entry scanner trades only from those
screened candidates for the day. For manual diagnostics after the entry window,
run `scripts/run_premarket_prefilter.py --ignore-entry-cutoff`.

CSV format:

```csv
symbol,period,strongBuy,buy,hold,sell,strongSell
AAPL,2026-06-01,12,20,8,1,0
```

## Paper Trading Soak Run

This project is profile-based:

- `paper` uses IB Gateway paper port `4002` and runtime folder `runtime/paper`.
- `live` uses IB Gateway live port `4001` and runtime folder `runtime/live`.

The separate runtime folders keep paper state, live state, logs, lock files, and
dashboard data isolated.

1. Install and configure IB Gateway.
   The current machine already has IB Gateway at:
   `/home/harika/Jts/ibgateway/1046/ibgateway`

2. IBC is installed locally at `/home/harika/ibc`.
   Configure the local IBC profile files with your IBKR login:

   - paper: `/home/harika/ibc/config.paper.ini`
   - live: `/home/harika/ibc/config.live.ini`

   Keep broker username/password in IBC or an OS secret store, not in this
   repository.

3. Local environment files:

   ```bash
   cp .env.paper.example .env.paper.local
   cp .env.live.example .env.live.local
   ```

   `.env.paper.local` is ready for paper mode. `.env.live.local` is present but
   live trading remains blocked until you deliberately set:

   ```text
   VELOCITY_LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_RISK
   ```

4. Start paper trading in `nohup` mode:

   ```bash
   nohup ./scripts/start_trader.sh paper > logs/paper_autotrader_stdout.log 2> logs/paper_autotrader_stderr.log &
   ```

5. Start the paper dashboard:

   ```bash
   nohup ./scripts/start_dashboard.sh paper > logs/paper_dashboard_stdout.log 2> logs/paper_dashboard_stderr.log &
   ```

6. Check paper status:

   ```bash
   ./scripts/check_runtime.sh paper
   ```

Live commands, after paper validation and explicit acknowledgement:

```bash
nohup ./scripts/start_trader.sh live > logs/live_autotrader_stdout.log 2> logs/live_autotrader_stderr.log &
nohup ./scripts/start_dashboard.sh live > logs/live_dashboard_stdout.log 2> logs/live_dashboard_stderr.log &
./scripts/check_runtime.sh live
```

Important: IBC can automate Gateway login dialogs, but IBKR may still require
two-factor approval or manual recovery after maintenance, password changes,
session conflicts, or account/security prompts. Do not assume a 15-20 day run is
hands-free until alerts and the health check have proven stable in paper.
