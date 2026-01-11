# funsa_v2/portfolio.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PortfolioConfig


def _cap_and_renormalize(w: pd.Series, max_w: float) -> pd.Series:
    """
    Iteratively cap weights at max_w and renormalize remaining mass.
    """
    w = w.copy().fillna(0.0)
    if w.empty:
        return w

    # normalize to 1
    s = float(w.sum())
    if s <= 0:
        return w * 0.0
    w = w / s

    for _ in range(50):
        over = w > max_w
        if not bool(over.any()):
            break
        w.loc[over] = max_w
        remaining = 1.0 - float(w.sum())
        if remaining <= 1e-12:
            break
        under = w < max_w
        if not bool(under.any()):
            break
        u_sum = float(w.loc[under].sum())
        if u_sum <= 0:
            w.loc[under] = w.loc[under] + remaining / float(under.sum())
        else:
            w.loc[under] = w.loc[under] + remaining * (w.loc[under] / u_sum)

    # final clamp
    w = w.clip(lower=0.0, upper=max_w)
    # renormalize if drift
    s2 = float(w.sum())
    if s2 > 0:
        w = w / s2
    return w


@dataclass
class PortfolioResult:
    target_weights: Dict[str, float]
    picks: List[str]
    cash_weight: float
    diagnostics: Dict[str, float]


class PortfolioConstructorV2:
    """
    Simple portfolio constructor:
    - select top-N by score
    - convert to positive weights
    - enforce min/max position weights
    - allow cash if exposure_scale < 1 or if not enough valid picks
    """

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg

    def build(
        self,
        scores: pd.Series,           # sorted desc (may include NaNs)
        regime: str,                 # not used yet (kept for future regime-specific rules)
        confidence: float,           # not used yet (kept for future)
        exposure_scale: float,       # 0..1 gross exposure target
        valid_mask_count: int,       # diagnostics / safety
    ) -> PortfolioResult:
        c = self.cfg

        if exposure_scale <= 0.0:
            return PortfolioResult({}, [], 1.0, {"reason_zero_exposure": 1.0})

        s = scores.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            cash = 1.0 if c.allow_cash else max(0.0, 1.0 - exposure_scale)
            return PortfolioResult({}, [], cash, {"reason_no_scores": 1.0, "valid_mask_count": float(valid_mask_count)})

        picks = list(s.head(int(c.max_positions)).index)

        # Convert scores to positive weights: shift to >=0, add epsilon
        base = s.loc[picks]
        if float(base.std(ddof=0)) == 0.0:
            w = pd.Series(1.0, index=picks)
        else:
            w = (base - float(base.min())) + 1e-6

        w = w / float(w.sum())

        # Apply min weight floor
        if c.min_weight and c.min_weight > 0:
            w = w.clip(lower=float(c.min_weight))
            w = w / float(w.sum())

        # Cap and renormalize
        w = _cap_and_renormalize(w, float(c.max_weight))

        # Scale by exposure
        w = w * float(exposure_scale)

        target = {k: float(v) for k, v in w.to_dict().items()}
        gross = float(sum(target.values()))
        cash_weight = max(0.0, 1.0 - gross) if c.allow_cash else max(0.0, 1.0 - exposure_scale)

        diag = {
            "gross_exposure": gross,
            "cash_weight": cash_weight,
            "max_positions": float(c.max_positions),
            "max_weight": float(c.max_weight),
            "min_weight": float(c.min_weight),
            "valid_mask_count": float(valid_mask_count),
        }

        return PortfolioResult(target, picks, cash_weight, diag)
