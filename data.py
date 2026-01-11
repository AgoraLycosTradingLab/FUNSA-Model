# funsa_v2/data.py
from __future__ import annotations

from typing import List

import pandas as pd

from .config import FUNSAv2Config

try:
    import yfinance as yf
except Exception:
    yf = None


class MarketData:
    """
    Config-driven yfinance loader.

    - fetch_ohlcv() returns MultiIndex columns: (field, ticker)
    - get_field() extracts a single field to a wide DF: index=date, columns=tickers
    """

    def __init__(self, cfg: FUNSAv2Config):
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        self.cfg = cfg

    def fetch_ohlcv(self, tickers: List[str]) -> pd.DataFrame:
        df = yf.download(
            tickers=tickers,
            start=self.cfg.backtest.start,
            end=self.cfg.backtest.end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # If yfinance returns single-level columns (common for one ticker),
        # promote to MultiIndex (field, ticker)
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([df.columns, tickers])

        return df.sort_index()

    @staticmethod
    def get_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
        """
        Extract a single field (e.g., 'Adj Close', 'Volume') to wide dataframe (date x tickers).
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns (field, ticker).")

        available = set(df.columns.get_level_values(0))
        if field not in available:
            raise KeyError(f"Field '{field}' not found. Available: {sorted(available)}")

        return df[field].copy().sort_index()
