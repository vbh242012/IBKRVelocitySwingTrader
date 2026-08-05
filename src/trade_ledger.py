"""Per-trade measurement ledger.

Records every live round trip so win rate, expectancy, MFE/MAE, and
exit-reason attribution can be computed from real fills instead of log
forensics.  This closes the observability gap where broker-side TRAIL fills
left no exit record at all and the dashboard had no trade history.

Design constraints:

- Fail-soft.  The ledger must never raise into the trading path.  Engine code
  only calls it through ``VelocityEngineBase._ledger_call``, which guards
  every call, and this module additionally sidelines a corrupt file instead
  of crashing on load.
- Append-only history.  Closed trades are never rewritten; the file is the
  audit trail.
- Atomic persistence.  tmp + ``os.replace`` like the other runtime state
  files, so a crash mid-write cannot corrupt existing history.
- Honest measurement.  MFE/MAE are cycle-sampled (~1/min) from the engine's
  fresh-price refresh, so they slightly understate true intraday extremes.
  They are directional measurements, not tick-accurate ones.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Mapping, Optional

import numpy as np

from src.config import TRADE_LEDGER_FILE, TZ_ET

LEDGER_VERSION = 1

# Entry-context fields copied from engine state / scan context into the open
# record.  Whitelisted so runtime-only keys (pending flags, protection audit
# timestamps, cached prices) do not bloat the permanent file.
_ENTRY_FIELDS = (
    'score',
    'entry_strategy',
    'strategy_profile',
    'regime',
    'trailing_percent',
    'initial_stop_loss',
    'stop_dist',
    'spread_pct',
    'volume_pace',
    'atr_pct',
    'relative_strength_63d',
    'relative_strength_126d',
    'return_13w',
    'return_26w',
    'price_vs_52w_high',
    'weekly_uptrend',
    'analyst_rating_score',
    'analyst_rating_total',
)


def _now_iso() -> str:
    return datetime.now(TZ_ET).isoformat()


def _finite(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _trading_days_between(start_iso, end_iso) -> Optional[int]:
    try:
        start = datetime.fromisoformat(str(start_iso)).date()
        end = datetime.fromisoformat(str(end_iso)).date()
        return int(np.busday_count(start, end))
    except (TypeError, ValueError):
        return None


class TradeLedger:
    """One open record per held symbol; closed records appended forever."""

    def __init__(self, path: str = TRADE_LEDGER_FILE):
        self._path = path
        self._open: dict[str, dict] = {}
        self._closed: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("ledger root is not an object")
            self._open = {
                str(sym): dict(rec)
                for sym, rec in (payload.get('open') or {}).items()
                if isinstance(rec, dict)
            }
            self._closed = [
                dict(rec) for rec in (payload.get('closed') or [])
                if isinstance(rec, dict)
            ]
        except Exception:
            # A corrupt ledger must not block trading and must not be silently
            # overwritten either — sideline it for forensics and start fresh.
            sidecar = (
                f"{self._path}.corrupt-"
                f"{datetime.now(TZ_ET).strftime('%Y%m%d-%H%M%S')}"
            )
            try:
                os.replace(self._path, sidecar)
            except OSError:
                pass
            self._open = {}
            self._closed = []

    def _save(self) -> None:
        payload = {
            'version': LEDGER_VERSION,
            'updated_at': _now_iso(),
            'open': self._open,
            'closed': self._closed,
        }
        tmp = self._path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=1, default=str)
        os.replace(tmp, self._path)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def has_open(self, symbol: str) -> bool:
        return symbol in self._open

    def open_trade(self, symbol: str, record: Mapping, *, replace: bool = True) -> None:
        """Record a new open position.

        ``replace=True`` (fresh entry): an unexpected existing open record is
        closed as superseded so its history is not lost.
        ``replace=False`` (position recovered from broker after restart): an
        existing open record is the surviving truth — keep it untouched.
        """
        if symbol in self._open:
            if not replace:
                return
            self.close_trade(
                symbol,
                exit_price=None,
                exit_reason='superseded_by_new_entry',
                exit_price_source='none',
            )

        entry_price = _finite(record.get('fill_price')) or _finite(record.get('price')) or 0.0
        rec = {
            'ledger_version': LEDGER_VERSION,
            'symbol': symbol,
            'status': 'open',
            'source': record.get('source', 'entry'),
            'entry_time': record.get('time') or _now_iso(),
            'entry_price': round(entry_price, 4),
            'qty': _finite(record.get('qty')) or 0.0,
            'entry_order_id': record.get('entry_order_id'),
            'entry_commission': _finite(record.get('commission')),
            'mfe_price': round(entry_price, 4),
            'mae_price': round(entry_price, 4),
        }
        for field in _ENTRY_FIELDS:
            if field in record:
                rec[field] = record.get(field)
        self._open[symbol] = rec
        self._save()

    def update_price(self, symbol: str, price) -> None:
        """Cycle-sampled excursion tracking for the open record."""
        rec = self._open.get(symbol)
        px = _finite(price)
        if rec is None or px is None or px <= 0:
            return
        changed = False
        current_mfe = _finite(rec.get('mfe_price'))
        if current_mfe is None or px > current_mfe:
            rec['mfe_price'] = round(px, 4)
            changed = True
        current_mae = _finite(rec.get('mae_price'))
        if current_mae is None or px < current_mae:
            rec['mae_price'] = round(px, 4)
            changed = True
        if changed:
            self._save()

    def close_trade(
        self,
        symbol: str,
        *,
        exit_price,
        exit_reason: str,
        exit_time: Optional[str] = None,
        exit_commission=None,
        entry_commission=None,
        exit_price_source: str = 'software_fill',
        qty=None,
    ) -> None:
        """Finalize a round trip and append it to the immutable history."""
        rec = self._open.pop(symbol, None)
        if rec is None:
            # An exit must never be lost just because the entry was not
            # recorded (e.g. ledger introduced mid-position).
            rec = {
                'ledger_version': LEDGER_VERSION,
                'symbol': symbol,
                'status': 'open',
                'source': 'unknown',
                'entry_time': None,
                'entry_price': 0.0,
                'qty': 0.0,
                'note': 'closed_without_open_record',
            }

        exit_px = _finite(exit_price)
        entry_px = _finite(rec.get('entry_price')) or 0.0
        trade_qty = _finite(qty)
        if trade_qty is None or trade_qty <= 0:
            trade_qty = _finite(rec.get('qty')) or 0.0
        entry_comm = _finite(entry_commission)
        if entry_comm is None:
            entry_comm = _finite(rec.get('entry_commission'))
        exit_comm = _finite(exit_commission)

        rec['status'] = 'closed'
        rec['exit_time'] = exit_time or _now_iso()
        rec['exit_price'] = round(exit_px, 4) if exit_px is not None else None
        rec['exit_reason'] = str(exit_reason or 'unknown')
        rec['exit_price_source'] = exit_price_source
        rec['entry_commission'] = entry_comm
        rec['exit_commission'] = exit_comm
        rec['commissions_complete'] = entry_comm is not None and exit_comm is not None
        rec['qty'] = trade_qty

        if exit_px is not None:
            # Exit itself is an excursion sample.
            if exit_px > (_finite(rec.get('mfe_price')) or 0.0):
                rec['mfe_price'] = round(exit_px, 4)
            mae = _finite(rec.get('mae_price'))
            if mae is None or exit_px < mae:
                rec['mae_price'] = round(exit_px, 4)

        if exit_px is not None and entry_px > 0 and trade_qty > 0:
            gross = (exit_px - entry_px) * trade_qty
            rec['gross_pnl'] = round(gross, 2)
            rec['net_pnl'] = round(gross - (entry_comm or 0.0) - (exit_comm or 0.0), 2)
            rec['pnl_pct'] = round((exit_px - entry_px) / entry_px * 100, 3)
        else:
            rec['gross_pnl'] = None
            rec['net_pnl'] = None
            rec['pnl_pct'] = None

        mfe = _finite(rec.get('mfe_price'))
        mae = _finite(rec.get('mae_price'))
        if entry_px > 0:
            rec['mfe_pct'] = round((mfe - entry_px) / entry_px * 100, 3) if mfe is not None else None
            rec['mae_pct'] = round((mae - entry_px) / entry_px * 100, 3) if mae is not None else None
        rec['trading_days_held'] = _trading_days_between(
            rec.get('entry_time'), rec['exit_time']
        )

        self._closed.append(rec)
        self._save()

    # ── Reporting ─────────────────────────────────────────────────────────────
    def recent_closed(self, limit: int = 20) -> list[dict]:
        return [dict(rec) for rec in self._closed[-max(int(limit), 0):]]

    def summary(self) -> dict:
        pnls = [
            _finite(rec.get('net_pnl'))
            for rec in self._closed
            if _finite(rec.get('net_pnl')) is not None
        ]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            'closed_trades': len(self._closed),
            'measured_trades': len(pnls),
            'open_trades': len(self._open),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate_pct': round(len(wins) / len(pnls) * 100, 1) if pnls else None,
            'total_net_pnl': round(sum(pnls), 2) if pnls else 0.0,
            'avg_win': round(sum(wins) / len(wins), 2) if wins else None,
            'avg_loss': round(sum(losses) / len(losses), 2) if losses else None,
        }
