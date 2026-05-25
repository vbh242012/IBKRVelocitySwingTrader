# IBKRVelocitySwingTrader
Automated Swing Trading For Small Cash Account With T+1 Settlement Days Using Interactive Broker

## Paper Trading Soak Run

This project is configured to run against the IBKR paper gateway by default.
Use IBC as the external login/startup supervisor and let the trading app connect
only after the IB API socket is reachable.

1. Install and configure IB Gateway.
   The current machine already has IB Gateway at:
   `/home/harika/Jts/ibgateway/1046/ibgateway`

2. IBC is installed locally at `/home/harika/ibc`.
   Configure `/home/harika/ibc/config.ini` with your IBKR paper login.
   Keep broker username/password in IBC or an OS secret store, not in this
   repository.

3. The local paper environment file has been created at `.env.paper.local`.
   It points the app to `/home/harika/ibc/gatewaystart.sh -inline`.
   To recreate it later, copy the template:

   ```bash
   cp .env.paper.example .env.paper.local
   ```

4. Start the paper trader in `nohup` mode:

   ```bash
   nohup ./scripts/start_paper_trader.sh > logs/autotrader_stdout.log 2> logs/autotrader_stderr.log &
   ```

5. Start the dashboard:

   ```bash
   nohup ./scripts/start_dashboard.sh > logs/dashboard_stdout.log 2> logs/dashboard_stderr.log &
   ```

6. Check status every few days:

   ```bash
   ./scripts/check_paper_runtime.sh
   ```

Important: IBC can automate Gateway login dialogs, but IBKR may still require
two-factor approval or manual recovery after maintenance, password changes,
session conflicts, or account/security prompts. Do not assume a 15-20 day run is
hands-free until alerts and the health check have proven stable in paper.
