# funsa_v2/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# =========================
# Backtest / Runtime
# =========================

@dataclass
class BacktestConfig:
    start: str = "2018-01-01"
    end: str | None = None
    initial_capital: float = 100_000.0
    trading_cost_bps: float = 5.0  # round-trip assumption


# =========================
# Regime Gate (v2)
# =========================

@dataclass
class RegimeConfig:
    # tickers
    vix_ticker: str = "^VIX"
    rates_ticker: str = "TLT"
    credit_risk: str = "HYG"
    credit_safe: str = "LQD"

    # signal construction
    z_lookback: int = 252
    ema_fast: int = 20
    ema_slow: int = 63
    vix_trend_days: int = 5

    # component weighting
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "vix_level": 0.30,
        "vix_trend": 0.20,
        "credit": 0.30,
        "rates": 0.20,
    })

    # regime thresholds
    risk_on_min: float = 0.60
    risk_off_max: float = 0.40

    # exposure by regime
    exposure_risk_on: float = 1.00
    exposure_neutral: float = 0.50
    exposure_risk_off: float = 0.00

    # macro override
    cape_overvalued: bool = False


# =========================
# Signal Engine
# =========================

@dataclass
class SignalConfig:
    price_field: str = "Adj Close"

    # momentum / trend
    mom_20: int = 20
    mom_63: int = 63
    trend_ma_126: int = 126

    # volatility normalization
    vol_lookback: int = 20


# =========================
# Filters
# =========================

@dataclass
class FilterConfig:
    trend_ma_days: int = 200
    rvol_lookback: int = 20
    rvol_min: float = 1.0
    min_price: float = 5.0


# =========================
# Scoring
# =========================

@dataclass
class ScoringConfig:
    # base factor weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.60,
        "trend": 0.40,
    })

    # regime tilt multipliers
    regime_tilts: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RISK_ON": {"momentum": 1.2, "trend": 1.0},
        "NEUTRAL": {"momentum": 1.0, "trend": 1.0},
        "RISK_OFF": {"momentum": 0.6, "trend": 1.2},
    })


# =========================
# Portfolio Construction
# =========================

@dataclass
class PortfolioConfig:
    max_positions: int = 4
    max_weight: float = 0.30
    min_weight: float = 0.05
    rebalance_frequency: str = "W-FRI"
    allow_cash: bool = True


# =========================
# Risk Management
# =========================

@dataclass
class RiskConfig:
    stop_loss_pct: float = 0.10
    trailing_stop_pct: float = 0.15
    portfolio_kill_dd: float = 0.20
    max_turnover: float = 0.50


# =========================
# Master Config
# =========================

@dataclass
class FUNSAv2Config:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # universe
    universe_tickers: List[str] = field(default_factory=lambda: [
        # Regime tickers (must be included)
        "^VIX", "TLT", "HYG", "LQD",

        # Core ETFs (example)
        "XLI", "XLP", "XLY", "XLV", "XLK",
        "XLE", "XLF", "XLU", "XLB", "XLRE",
    ])
