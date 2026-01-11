# funsa_v2/backtest.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import FUNSAv2Config
from .data import MarketData
from .filters import FiltersV2
from .portfolio import PortfolioConstructorV2
from .regime import RegimeGate
from .risk import RiskManagerV2, RiskState
from .scoring import ScoringV2
from .signals import SignalEngineV2


@dataclass
class LiveSignal:
    active: bool
    regime: object
    picks: List[str]
    target_weights: Dict[str, float]
    cash_weight: float
    scores: Dict[str, float]
    diagnostics: Dict[str, object]


class FUNSAv2Model:
    """
    FUNSA v2 orchestration:
      data -> regime gate -> signals -> filters -> scoring -> portfolio -> diagnostics
    """

    def __init__(self, cfg: FUNSAv2Config):
        self.cfg = cfg

        # Data is config-driven (no DataSpec)
        self.data = MarketData(cfg)

        self.regime = RegimeGate(cfg)
        self.signals = SignalEngineV2(cfg.signals)
        self.filters = FiltersV2(cfg.filters)
        self.scoring = ScoringV2(cfg.scoring, cfg.filters)
        self.portfolio = PortfolioConstructorV2(cfg.portfolio)
        self.risk = RiskManagerV2(cfg.risk)

    def _split_inputs(self, prices: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        rg = self.cfg.regime
        need = [rg.vix_ticker, rg.rates_ticker, rg.credit_risk, rg.credit_safe]

        missing = [t for t in need if t not in prices.columns]
        reg_closes = pd.DataFrame() if missing else prices[need].dropna(how="all")

        trade_tickers = [t for t in prices.columns if t not in set(need)]
        trade_prices = prices[trade_tickers].dropna(how="all")
        trade_volume = volume[trade_tickers].reindex(trade_prices.index).fillna(0.0)

        return {
            "reg_closes": reg_closes,
            "trade_prices": trade_prices,
            "trade_volume": trade_volume,
            "reg_missing": pd.DataFrame({"missing": missing}) if missing else pd.DataFrame(),
        }

    def generate_live_signal(self, debug_imports: bool = False) -> LiveSignal:
        tickers = list(dict.fromkeys(self.cfg.universe_tickers))
        raw = self.data.fetch_ohlcv(tickers)
        if raw.empty:
            return LiveSignal(False, None, [], {}, 1.0, {}, {"reason": "no_data"})

        prices = self.data.get_field(raw, self.cfg.signals.price_field).dropna(how="all")
        volume = self.data.get_field(raw, "Volume").reindex(prices.index).fillna(0.0)

        parts = self._split_inputs(prices, volume)
        reg_closes = parts["reg_closes"]
        px = parts["trade_prices"]
        vol = parts["trade_volume"]
        missing = list(parts["reg_missing"].get("missing", []))

        if reg_closes.empty:
            return LiveSignal(False, None, [], {}, 1.0, {}, {"reason": "missing_regime_series", "missing": missing})
        if px.empty:
            return LiveSignal(False, None, [], {}, 1.0, {}, {"reason": "missing_trade_series"})

        # --- Regime (v2) ---
        reg = self.regime.compute(reg_closes)

        dbg = {}
        if debug_imports:
            import inspect
            dbg = {
                "regime_module": inspect.getfile(self.regime.__class__),
                "regime_class": str(self.regime.__class__),
                "regime_state_fields": getattr(reg, "__dict__", None),
            }

        if (not reg.active) or (reg.exposure_scale <= 0):
            return LiveSignal(False, reg, [], {}, 1.0, {}, {"reason": "inactive_or_zero_exposure", **dbg})

        # --- Signals ---
        raw_sig = self.signals.compute_raw(px)
        norm = self.signals.normalize_latest(raw_sig)

        # --- Filters ---
        valid_mask = self.filters.compute_mask(px, vol, raw_sig["vol_20"])
        comps = self.filters.compute_components(px, vol, raw_sig["vol_20"])

        # --- Scoring ---
        score_res = self.scoring.score(norm, valid_mask, reg.label, reg.confidence)

        # --- Portfolio ---
        port_res = self.portfolio.build(
            scores=score_res.scores,
            regime=reg.label,
            confidence=reg.confidence,
            exposure_scale=reg.exposure_scale,
            valid_mask_count=int(valid_mask.sum()),
        )

        picks = port_res.picks
        scores_dict = {t: float(score_res.scores.loc[t]) for t in picks if t in score_res.scores.index}

        diagnostics = {
            "risk_score": float(reg.risk_score),
            "confidence": float(reg.confidence),
            "exposure_scale": float(reg.exposure_scale),
            "regime_components_01": reg.components_01,
            "regime_raw_z": reg.raw_z,
            "filters": {
                "passed": int(valid_mask.sum()),
                "failed": int((~valid_mask).sum()),
                "rvol_top5": comps["rvol"].dropna().sort_values(ascending=False).head(5).to_dict(),
            },
            "scoring": score_res.diagnostics,
            "portfolio": port_res.diagnostics,
            **dbg,
        }

        return LiveSignal(
            active=True,
            regime=reg,
            picks=picks,
            target_weights=port_res.target_weights,
            cash_weight=port_res.cash_weight,
            scores=scores_dict,
            diagnostics=diagnostics,
        )

    def run_backtest(self) -> pd.DataFrame:
        tickers = list(dict.fromkeys(self.cfg.universe_tickers))
        raw = self.data.fetch_ohlcv(tickers)
        if raw.empty:
            return pd.DataFrame()

        prices = self.data.get_field(raw, self.cfg.signals.price_field).dropna(how="all")
        volume = self.data.get_field(raw, "Volume").reindex(prices.index).fillna(0.0)

        parts = self._split_inputs(prices, volume)
        reg_closes = parts["reg_closes"]
        px = parts["trade_prices"]
        vol = parts["trade_volume"]

        if reg_closes.empty or px.empty:
            return pd.DataFrame()

        # Precompute trade signals over time
        raw_sig_all = self.signals.compute_raw(px)

        # Rebalance dates
        rebal_dates = pd.date_range(px.index.min(), px.index.max(), freq=self.cfg.portfolio.rebalance_frequency)
        rebal_dates = rebal_dates.intersection(px.index)

        equity = float(self.cfg.backtest.initial_capital)
        cash = equity
        holdings: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        entry_prices: Dict[str, float] = {}
        risk_state = RiskState(peak_equity=equity, kill_switch=False)

        rows = []

        min_hist = max(
            self.cfg.signals.mom_20,
            self.cfg.signals.trend_ma_126,
            self.cfg.filters.trend_ma_days,
            self.cfg.filters.rvol_lookback,
            self.cfg.signals.vol_lookback,
        ) + 5

        for dt in px.index:
            # Mark-to-market
            mkt_value = 0.0
            for t, sh in holdings.items():
                if t in px.columns:
                    p = float(px.loc[dt, t])
                    if not np.isnan(p):
                        mkt_value += sh * p
            equity = cash + mkt_value

            # Kill switch update
            risk_state = self.risk.update_kill_switch(risk_state, equity)

            if risk_state.kill_switch:
                holdings, weights, entry_prices = {}, {}, {}
                cash = equity
            else:
                # Position stops daily
                stopped_weights = self.risk.apply_position_stops(entry_prices, px.loc[dt], weights)
                if set(stopped_weights.keys()) != set(weights.keys()):
                    for t in list(holdings.keys()):
                        if t not in stopped_weights:
                            p = float(px.loc[dt, t])
                            cash += holdings[t] * p
                            holdings.pop(t, None)
                            entry_prices.pop(t, None)
                    weights = stopped_weights

            # Rebalance
            if (dt in rebal_dates) and (not risk_state.kill_switch):
                if len(px.loc[:dt]) >= min_hist and len(reg_closes.loc[:dt]) >= min_hist:
                    reg = self.regime.compute(reg_closes.loc[:dt])

                    if not reg.active or reg.exposure_scale <= 0:
                        holdings, weights, entry_prices = {}, {}, {}
                        cash = equity
                    else:
                        latest_raw = {k: v.loc[:dt] for k, v in raw_sig_all.items()}
                        norm = self.signals.normalize_latest(latest_raw)

                        valid_mask = self.filters.compute_mask(px.loc[:dt], vol.loc[:dt], latest_raw["vol_20"])
                        score_res = self.scoring.score(norm, valid_mask, reg.label, reg.confidence)

                        port_res = self.portfolio.build(
                            scores=score_res.scores,
                            regime=reg.label,
                            confidence=reg.confidence,
                            exposure_scale=reg.exposure_scale,
                            valid_mask_count=int(valid_mask.sum()),
                        )
                        w_new = dict(port_res.target_weights)

                        # Turnover guardrail
                        if self.risk.should_skip_rebalance(weights, w_new):
                            w_new = weights

                        # Liquidate removed names
                        for t in list(holdings.keys()):
                            if t not in w_new:
                                p = float(px.loc[dt, t])
                                cash += holdings[t] * p
                                holdings.pop(t, None)
                                entry_prices.pop(t, None)

                        # Recompute equity after liquidations
                        equity = cash + sum(holdings.get(t, 0.0) * float(px.loc[dt, t]) for t in holdings)

                        # Allocate to targets (continuous shares)
                        for t, w in w_new.items():
                            if t not in px.columns:
                                continue
                            p = float(px.loc[dt, t])
                            if p <= 0 or np.isnan(p):
                                continue

                            target_value = equity * float(w)
                            current_value = holdings.get(t, 0.0) * p
                            delta_value = target_value - current_value

                            # Trading costs
                            cost = abs(delta_value) * (self.cfg.backtest.trading_cost_bps / 10_000.0)
                            cash -= cost

                            delta_shares = delta_value / p
                            holdings[t] = holdings.get(t, 0.0) + delta_shares
                            cash -= delta_shares * p

                            # Entry price for stops
                            if t not in entry_prices or abs(current_value) < 1e-6:
                                entry_prices[t] = p

                        weights = dict(w_new)

            # Record
            pos_val = {t: holdings.get(t, 0.0) * float(px.loc[dt, t]) for t in px.columns}
            gross = float(sum(abs(v) for v in pos_val.values()))
            net = float(sum(v for v in pos_val.values()))

            rows.append({
                "date": dt,
                "equity": float(equity),
                "cash": float(cash),
                "gross_exposure": gross / equity if equity > 0 else 0.0,
                "net_exposure": net / equity if equity > 0 else 0.0,
                "kill_switch": bool(risk_state.kill_switch),
                "positions": ",".join(sorted([t for t in holdings if abs(holdings[t]) > 1e-8])),
            })

        out = pd.DataFrame(rows).set_index("date")
        out["returns"] = out["equity"].pct_change().fillna(0.0)
        return out


def summarize_backtest(bt: pd.DataFrame) -> Dict[str, float]:
    if bt is None or bt.empty:
        return {}

    cagr = (bt["equity"].iloc[-1] / bt["equity"].iloc[0]) ** (252 / max(len(bt), 1)) - 1
    sharpe = (bt["returns"].mean() / (bt["returns"].std(ddof=0) + 1e-12)) * math.sqrt(252)
    peak = bt["equity"].cummax()
    dd = (peak - bt["equity"]) / peak

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.max()),
        "final_equity": float(bt["equity"].iloc[-1]),
    }
