# FUNSA-Model
FUNSA is a regime-aware capital allocation model. It evaluates macro risk (volatility, credit, rates) to determine market regime, confidence, and allowable exposure. Output specifies how much capital to deploy, how much to hold in cash, and which sectors to allocate to.

FUNSA v2 — purpose
Role: Capital allocator + exposure governor

What it does:
  Takes macro inputs (vol, credit, rates)
  
  Produces continuous risk measures:
    1.risk_score (0–1)
    2.confidence (0–1)
    3.exposure_scale (how much capital to deploy)

  Allocates capital across:
    1.sectors (ETFs)
    2.cash

  Enforces:
    1.cash buffers
    2.exposure throttling
    3.regime-aware positioning

  What it does not do:
    1.It does not pick individual stocks
    2.It does not design factor signals
    3.It does not try to maximize returns in isolation

Think of FUNSA as:
  A risk governor and capital throttle

Example output meaning:
  “Market is mildly favorable but low conviction → deploy ~30%, keep 70% cash.”

Below is the difference between the Funsa Model and the Alpha R1 Macro Model in the Regime Sniper Stack
| Dimension                   | Alpha R1 Macro        | FUNSA v2           |
| --------------------------- | --------------------- | ------------------ |
| Primary role                | Market diagnosis      | Capital allocation |
| Output type                 | Regime + factor gates | Weights + cash     |
| Exposure control            | ❌ No                  | ✅ Yes              |
| Cash management             | ❌ No                  | ✅ Yes              |
| Factor awareness            | ✅ Strong              | ❌ Minimal          |
| Sector rotation             | ❌ No                  | ✅ Yes              |
| Continuous risk scale       | ❌ Mostly discrete     | ✅ Continuous       |
| Designed to sit above alpha | ⚠️ Partial            | ✅ Yes              |


