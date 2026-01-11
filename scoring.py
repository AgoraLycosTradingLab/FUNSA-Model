# funsa_v2/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import FilterConfig, ScoringConfig


@dataclass
class ScoreResult:
    scores: pd.Series
    diagnostics: Dict[str, float]


class ScoringV2:
    """
    Combine normalized signals into a single score, with regime tilts and confidence blending.

    Inputs:
      norm_signals: dict[str, pd.Series] (latest-only, indexed by tickers)
      valid_mask: pd.Series[bool] indexed by tickers
      regime: "RISK_ON" | "NEUTRAL" | "RISK_OFF"
      confidence: 0..1
    """

    def __init__(self, cfg: ScoringConfig, filter_cfg: FilterConfig):
        self.cfg = cfg
        self.filter_cfg = filter_cfg

    def score(
        self,
        norm_signals: Dict[str, pd.Series],
        valid_mask: pd.Series,
        regime: str,
        confidence: float,
    ) -> ScoreResult:
        c = self.cfg
        conf = float(np.clip(confidence, 0.0, 1.0))

        # Expected signal keys from SignalEngineV2.normalize_latest()
        mom = norm_signals.get("momentum", pd.Series(dtype=float)).copy()
        trd = norm_signals.get("trend", pd.Series(dtype=float)).copy()

        # Align universe
        idx = valid_mask.index
        mom = mom.reindex(idx).astype(float)
        trd = trd.reindex(idx).astype(float)

        # Apply eligibility mask
        mom = mom.where(valid_mask, np.nan)
        trd = trd.where(valid_mask, np.nan)

        base_w = c.weights
        base = (
            float(base_w.get("momentum", 0.0)) * mom +
            float(base_w.get("trend", 0.0)) * trd
        )

        # Regime tilts
        tilts = c.regime_tilts.get(regime, {})
        tilt_m = float(tilts.get("momentum", 1.0))
        tilt_t = float(tilts.get("trend", 1.0))

        tilted = (
            float(base_w.get("momentum", 0.0)) * tilt_m * mom +
            float(base_w.get("trend", 0.0)) * tilt_t * trd
        )

        # Confidence blend between base and tilted
        scores = (1.0 - conf) * base + conf * tilted

        # Center/scale lightly (optional): convert NaNs to drop later
        scores = scores.replace([np.inf, -np.inf], np.nan)

        # Diagnostics
        s_valid = scores.dropna()
        dispersion = float(s_valid.std(ddof=0)) if len(s_valid) > 1 else 0.0
        top = float(s_valid.max()) if len(s_valid) else float("nan")
        bot = float(s_valid.min()) if len(s_valid) else float("nan")

        diag = {
            "n_valid": float(valid_mask.sum()),
            "confidence": conf,
            "dispersion": dispersion,
            "top_score": top,
            "bottom_score": bot,
            "tilt_momentum": tilt_m,
            "tilt_trend": tilt_t,
        }

        # Sort descending for downstream selection
        scores = scores.sort_values(ascending=False)

        return ScoreResult(scores=scores, diagnostics=diag)
