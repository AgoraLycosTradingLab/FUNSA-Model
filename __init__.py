# funsa_v2/__init__.py
"""
FUNSA v2 package.

Keep this file lightweight to avoid circular imports and import-time failures.
Do not import submodules here.
"""

__all__ = [
    "config",
    "data",
    "regime",
    "signals",
    "filters",
    "scoring",
    "portfolio",
    "risk",
    "backtest",
]
