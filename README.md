# IBKRVelocitySwingTrader
Automated Swing Trading For Small Cash Account With T+1 Settlement Days Using Interactive Broker

## Strategy

The default profile is now `indicator_swing`.

The bot is a short swing system. The old ORB/legacy strategy profiles have
been removed; `indicator_swing` is the only maintained strategy profile:

- The default profile is relative-strength first. A stock must pass trend,
  liquidity, relative-strength, weekly-uptrend, MA50/MA200, 52-week-high
  proximity, ATR, spread, and volume gates before an indicator sleeve can buy.
- EMA20 above SMA50 is the default trend state; fresh crosses, MA reclaims, or
  prior-high breaks can time entries after the RS gate passes. Bollinger lower
  band reclaim and PSAR flip are optional `indicator_swing` sleeves, but they
  are not enabled by default because multi-year validation did not justify
  using them as primary live triggers.
- RSI momentum/recovery plus at least two of MACD, stochastic, OBV, PSAR, and
  volume pace must confirm. Analyst ratings can adjust score, but cannot create
  a buy by themselves. The default minimum entry score is 50.
- Each position stores the sleeve that opened it and exits on that same sleeve's
  sell rule.
- Swing exits avoid same-day EOD churn: winners trim nearest whole-share
  cumulative profit tiers at +1R, +1.5R, and +2R so roughly 20%, 40%, and 60%
  of the original position has been sold. `R` is the original per-share
  Chandelier risk distance captured at entry. The remaining runner is protected
  by the broker Chandelier stop; software hard stop, close-confirmed break-even
  protection after +1R/first tier, analyst downgrade only with price weakness,
  and a 10-bar no-progress time stop remain active safety exits. Position size
  is based on the broker-protected Chandelier distance.

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
