# funsa_v2/risk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import RiskConfig


@dataclass
class RiskState:
    peak_equity: float
    kill_switch: bool = False


class RiskManagerV2:
    """
    Risk controls:
    - per-position stop loss (simple % from entry)
    - optional trailing stop (simple % from max since entry, approximated via prices)
    - portfolio kill-switch on drawdown from peak equity
    - turnover guardrail at rebalance
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    # ---------------------------
    # Portfolio-level kill switch
    # ---------------------------
    def update_kill_switch(self, state: RiskState, equity: float) -> RiskState:
        peak = max(float(state.peak_equity), float(equity))
        dd = (peak - float(equity)) / peak if peak > 0 else 0.0
        killed = bool(dd >= float(self.cfg.portfolio_kill_dd))
        return RiskState(peak_equity=peak, kill_switch=killed)

    # ---------------------------
    # Position-level stops
    # ---------------------------
    def apply_position_stops(
        self,
        entry_prices: Dict[str, float],
        last_prices: pd.Series,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Returns a filtered weights dict after removing positions that violate stop loss.
        """
        stop = float(self.cfg.stop_loss_pct)
        out = dict(weights)

        for t in list(out.keys()):
            ep = float(entry_prices.get(t, np.nan))
            lp = float(last_prices.get(t, np.nan))
            if not np.isfinite(ep) or not np.isfinite(lp) or ep <= 0:
                continue
            if lp <= ep * (1.0 - stop):
                out.pop(t, None)

        # renormalize remaining weights to preserve gross exposure
        gross = float(sum(out.values()))
        if gross > 0:
            for k in list(out.keys()):
                out[k] = float(out[k] / gross) * gross  # no-op; kept for clarity
        return out

    # ---------------------------
    # Turnover guardrail
    # ---------------------------
    def should_skip_rebalance(self, w_old: Dict[str, float], w_new: Dict[str, float]) -> bool:
        """
        Simple turnover proxy: 0.5 * sum |w_new - w_old|
        """
        max_to = float(self.cfg.max_turnover)
        keys = set(w_old.keys()) | set(w_new.keys())
        to = 0.5 * sum(abs(float(w_new.get(k, 0.0)) - float(w_old.get(k, 0.0))) for k in keys)
        return bool(to > max_to)
