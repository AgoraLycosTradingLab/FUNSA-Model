# run_funsa_v2.py
"""
Runner for FUNSA v2 (clean rebuild).

Usage:
  python run_funsa_v2.py
"""

from __future__ import annotations

# DEV MODE HOT RELOAD (safe)
import dev_reload
dev_reload.reload_funsa_v2()

import math
import inspect

from funsa_v2.config import FUNSAv2Config
from funsa_v2.backtest import FUNSAv2Model



def main() -> None:
    # --- verify import paths (one-time sanity) ---
    print("BACKTEST FILE:", inspect.getfile(FUNSAv2Model))

    cfg = FUNSAv2Config()
    model = FUNSAv2Model(cfg)

    # =====================
    # LIVE SNAPSHOT
    # =====================
    live = model.generate_live_signal(debug_imports=True)

    print("\n=== FUNSA v2 LIVE SIGNAL ===")
    print(f"Active: {live.active}")
    print(f"Regime: {live.regime}")
    print(f"Picks: {live.picks}")
    print(f"Target Weights: {live.target_weights}")
    print(f"Cash Weight: {live.cash_weight:.3f}")

    print("Diagnostics (high level):")
    if live.diagnostics:
        # import/path debug
        if live.diagnostics.get("regime_module"):
            print(f"  regime_module: {live.diagnostics.get('regime_module')}")
        if live.diagnostics.get("regime_class"):
            print(f"  regime_class: {live.diagnostics.get('regime_class')}")

        # regime metrics
        print(f"  vix: {getattr(live.regime, 'vix', None)}")
        print(f"  risk_score: {live.diagnostics.get('risk_score')}")
        print(f"  confidence: {live.diagnostics.get('confidence')}")
        print(f"  exposure_scale: {live.diagnostics.get('exposure_scale')}")

        # pipeline metrics
        print(f"  filters.passed: {live.diagnostics.get('filters', {}).get('passed')}")
        print(f"  scoring.dispersion: {live.diagnostics.get('scoring', {}).get('dispersion')}")

        # Uncomment for deep debug
        # print("  regime_components_01:", live.diagnostics.get("regime_components_01"))
        # print("  regime_raw_z:", live.diagnostics.get("regime_raw_z"))

    # =====================
    # BACKTEST
    # =====================
    bt = model.run_backtest()
    if bt is None or bt.empty:
        print("\nNo backtest data returned.")
        return

    # Basic stats
    cagr = (bt["equity"].iloc[-1] / bt["equity"].iloc[0]) ** (252 / max(len(bt), 1)) - 1
    sharpe = (bt["returns"].mean() / (bt["returns"].std(ddof=0) + 1e-12)) * math.sqrt(252)
    peak = bt["equity"].cummax()
    dd = (peak - bt["equity"]) / peak

    print("\n=== BACKTEST SUMMARY (basic) ===")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {dd.max():.2%}")
    print(f"Final Equity: {bt['equity'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
