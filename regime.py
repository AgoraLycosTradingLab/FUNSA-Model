# funsa_v2/regime.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import FUNSAv2Config


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def zscore_roll(series: pd.Series, lookback: int) -> pd.Series:
    # robust rolling z-score
    minp = max(20, lookback // 5)
    mu = series.rolling(lookback, min_periods=minp).mean()
    sd = series.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (series - mu) / sd


def sigmoid(x: float) -> float:
    # map real -> (0,1), tolerant to NaNs
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.5
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class RegimeState:
    label: str                  # "RISK_ON" | "NEUTRAL" | "RISK_OFF"
    vix: float                  # raw VIX level
    active: bool                # CAPE gate / kill switch can force False
    exposure_scale: float       # 0..1

    # v2 additions
    risk_score: float           # 0..1 (higher = more risk-on)
    confidence: float           # 0..1 (distance from neutral)
    components_01: Dict[str, float]  # per-component (0,1) scores
    raw_z: Dict[str, float]          # underlying z-scores


class RegimeGate:
    """
    FUNSA v2 Regime Gate:
    - Vol: VIX level + VIX short-term trend
    - Credit: trend of (HYG/LQD) ratio
    - Rates: trend of TLT

    Produces a continuous risk_score (0..1) and confidence (0..1).
    Exposure is base-by-regime * confidence throttle.

    Requires closes columns:
      cfg.regime.vix_ticker, cfg.regime.rates_ticker, cfg.regime.credit_risk, cfg.regime.credit_safe
    """

    def __init__(self, cfg: FUNSAv2Config):
        self.cfg = cfg

    def compute(self, closes: pd.DataFrame) -> RegimeState:
        c = self.cfg.regime

        px = closes.sort_index().ffill().dropna(how="all")
        for t in (c.vix_ticker, c.rates_ticker, c.credit_risk, c.credit_safe):
            if t not in px.columns:
                raise KeyError(f"RegimeGate missing required ticker in closes: {t}")

        vix = px[c.vix_ticker].dropna()
        tlt = px[c.rates_ticker].dropna()
        hyg = px[c.credit_risk].dropna()
        lqd = px[c.credit_safe].dropna()

        # align common index
        idx = vix.index.intersection(tlt.index).intersection(hyg.index).intersection(lqd.index)
        vix, tlt, hyg, lqd = vix.reindex(idx), tlt.reindex(idx), hyg.reindex(idx), lqd.reindex(idx)

        # z-scores
        vix_level_z = float(zscore_roll(vix, c.z_lookback).iloc[-1])
        vix_trend_z = float(zscore_roll(vix.pct_change(c.vix_trend_days), c.z_lookback).iloc[-1])

        credit_ratio = hyg / lqd
        credit_trend = ema(credit_ratio, c.ema_fast) / ema(credit_ratio, c.ema_slow) - 1.0
        credit_trend_z = float(zscore_roll(credit_trend, c.z_lookback).iloc[-1])

        rates_trend = ema(tlt, c.ema_fast) / ema(tlt, c.ema_slow) - 1.0
        rates_trend_z = float(zscore_roll(rates_trend, c.z_lookback).iloc[-1])

        raw_z = {
            "vix_level_z": vix_level_z,
            "vix_trend_z": vix_trend_z,
            "credit_trend_z": credit_trend_z,
            "rates_trend_z": rates_trend_z,
        }

        # map to (0,1) "ok-ness"
        comps = {
            "vix_level": sigmoid(-vix_level_z),     # high VIX => worse
            "vix_trend": sigmoid(-vix_trend_z),     # rising VIX => worse
            "credit": sigmoid(credit_trend_z),      # improving HY vs IG => better
            "rates": sigmoid(rates_trend_z),        # rising TLT => supportive
        }

        w = c.component_weights
        num = sum(float(w[k]) * float(comps[k]) for k in comps)
        den = sum(float(w[k]) for k in comps) or 1.0
        risk_score = float(num / den)

        # confidence: distance from neutral (0.5)
        confidence = float(2.0 * abs(risk_score - 0.5))

        # regime label + base exposure
        if risk_score > c.risk_on_min:
            label = "RISK_ON"
            base = c.exposure_risk_on
        elif risk_score < c.risk_off_max:
            label = "RISK_OFF"
            base = c.exposure_risk_off
        else:
            label = "NEUTRAL"
            base = c.exposure_neutral

        # CAPE gate
        active = not bool(c.cape_overvalued)
        if not active:
            exposure_scale = 0.0
        else:
            # confidence throttle: 0.5..1.0 multiplier
            exposure_scale = float(base * (0.5 + 0.5 * confidence))

        return RegimeState(
            label=label,
            vix=float(vix.iloc[-1]),
            active=active,
            exposure_scale=exposure_scale,
            risk_score=risk_score,
            confidence=confidence,
            components_01={k: float(v) for k, v in comps.items()},
            raw_z={k: float(v) for k, v in raw_z.items()},
        )
