# funsa_v2/signals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import SignalConfig


def _safe_zscore_cross_section(x: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score for a single date (Series over tickers).
    Returns NaN where input is NaN.
    """
    x = x.replace([np.inf, -np.inf], np.nan)
    mu = float(x.mean(skipna=True)) if x.notna().any() else 0.0
    sd = float(x.std(ddof=0, skipna=True)) if x.notna().any() else 0.0
    if sd <= 1e-12:
        return x * 0.0
    return (x - mu) / sd


class SignalEngineV2:
    """
    Multi-horizon signal builder:
      - momentum: blend of 20d and 63d returns
      - trend: distance above/below 126d MA
      - vol_20: rolling stdev of daily returns (for filters / scaling)

    compute_raw(prices) returns dict[str, DataFrame] time series of raw signals.
    normalize_latest(raw) returns dict[str, Series] for latest date (cross-sectional z-scores).
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def compute_raw(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        c = self.cfg
        px = prices.sort_index().ffill()

        ret1 = px.pct_change()

        mom20 = px.pct_change(int(c.mom_20))
        mom63 = px.pct_change(int(c.mom_63))
        # simple blend (equal weight); you can change later
        momentum = 0.5 * mom20 + 0.5 * mom63

        ma126 = px.rolling(int(c.trend_ma_126)).mean()
        trend = (px / ma126) - 1.0

        vol_20 = ret1.rolling(int(c.vol_lookback)).std(ddof=0)

        return {
            "momentum": momentum,
            "trend": trend,
            "vol_20": vol_20,
        }

    def normalize_latest(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Take the most recent row of each raw signal and z-score cross-sectionally.
        """
        out: Dict[str, pd.Series] = {}

        for key, df in raw.items():
            if df is None or df.empty:
                out[key] = pd.Series(dtype=float)
                continue

            last = df.iloc[-1].copy()
            out[key] = _safe_zscore_cross_section(last)

        return out
