# IBKRVelocitySwingTrader
Automated Swing Trading For Small Cash Account With T+1 Settlement Days Using Interactive Broker

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

Backward-compatible paper helpers still work:

```bash
./scripts/start_paper_trader.sh
./scripts/check_paper_runtime.sh
```

Important: IBC can automate Gateway login dialogs, but IBKR may still require
two-factor approval or manual recovery after maintenance, password changes,
session conflicts, or account/security prompts. Do not assume a 15-20 day run is
hands-free until alerts and the health check have proven stable in paper.
