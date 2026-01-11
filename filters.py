# funsa_v2/filters.py
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import FilterConfig


class FiltersV2:
    """
    Lightweight eligibility filters (boolean mask) + diagnostics.

    Inputs are wide DataFrames:
      prices: date x tickers
      volume: date x tickers
      vol_20: date x tickers (annualized or raw stdev; your signals decide)
    """

    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg

    def compute_components(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        vol_20: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        c = self.cfg

        px = prices.sort_index().ffill()
        vol = volume.sort_index().fillna(0.0)

        # --- price sanity ---
        last_px = px.iloc[-1].replace([np.inf, -np.inf], np.nan)
        price_ok = (last_px >= float(c.min_price)).fillna(False)

        # --- trend filter ---
        ma = px.rolling(int(c.trend_ma_days)).mean()
        trend_ok = (px.iloc[-1] > ma.iloc[-1]).fillna(False)

        # --- relative volume (RVOL) ---
        vol_ma = vol.rolling(int(c.rvol_lookback)).mean()
        rvol = (vol.iloc[-1] / vol_ma.iloc[-1]).replace([np.inf, -np.inf], np.nan)
        rvol_ok = (rvol >= float(c.rvol_min)).fillna(False)

        # --- volatility availability (optional gate; just ensure we have it) ---
        vol_latest = vol_20.iloc[-1].replace([np.inf, -np.inf], np.nan)
        vol_ok = vol_latest.notna()

        return {
            "price_ok": price_ok,
            "trend_ok": trend_ok,
            "rvol_ok": rvol_ok,
            "vol_ok": vol_ok,
            "rvol": rvol,
            "last_price": last_px,
            "vol_latest": vol_latest,
        }

    def compute_mask(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        vol_20: pd.DataFrame,
    ) -> pd.Series:
        comps = self.compute_components(prices, volume, vol_20)
        mask = (comps["price_ok"] & comps["trend_ok"] & comps["rvol_ok"] & comps["vol_ok"]).fillna(False)
        return mask
