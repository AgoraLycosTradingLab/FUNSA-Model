# debug_imports.py
"""
Minimal import debugger for FUNSA v2.

Run from the same directory as run_funsa_v2.py:
  python debug_imports.py
"""

import inspect

# Import submodules directly (avoid package-level side effects)
import funsa_v2.data as data
import funsa_v2.regime as regime
import funsa_v2.signals as signals
import funsa_v2.filters as filters
import funsa_v2.scoring as scoring
import funsa_v2.portfolio as portfolio
import funsa_v2.risk as risk
import funsa_v2.backtest as backtest

print("=== IMPORT PATHS ===")
print("data.py      :", inspect.getfile(data))
print("regime.py    :", inspect.getfile(regime))
print("signals.py   :", inspect.getfile(signals))
print("filters.py   :", inspect.getfile(filters))
print("scoring.py   :", inspect.getfile(scoring))
print("portfolio.py :", inspect.getfile(portfolio))
print("risk.py      :", inspect.getfile(risk))
print("backtest.py  :", inspect.getfile(backtest))

print("\n=== SYMBOL CHECKS ===")
print("MarketData in data:", hasattr(data, "MarketData"))
print("RegimeGate in regime:", hasattr(regime, "RegimeGate"))
print("SignalEngineV2 in signals:", hasattr(signals, "SignalEngineV2"))
print("FiltersV2 in filters:", hasattr(filters, "FiltersV2"))
print("ScoringV2 in scoring:", hasattr(scoring, "ScoringV2"))
print("PortfolioConstructorV2 in portfolio:", hasattr(portfolio, "PortfolioConstructorV2"))
print("RiskManagerV2 in risk:", hasattr(risk, "RiskManagerV2"))
print("FUNSAv2Model in backtest:", hasattr(backtest, "FUNSAv2Model"))

print("\n=== METHOD SIGNATURES ===")
print("generate_live_signal:",
      inspect.signature(backtest.FUNSAv2Model.generate_live_signal))
print("run_backtest:",
      inspect.signature(backtest.FUNSAv2Model.run_backtest))
