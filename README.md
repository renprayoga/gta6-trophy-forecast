# GTA VI Trophy Rarity Forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/renprayoga/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/renprayoga/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every GTA VI PlayStation 5 trophy, its Exophase earn rate
at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by mean pinball loss on the logit scale.

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic.
Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`/output/<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it forecasts

Per trophy, in logit space with `x = ln(1 + hours since release)`:

1. **Collapse to distinct snapshots.** Exophase refreshes a game's counts
   hours apart, so consecutive identical reads carry no information and
   would overweight stale periods in any fit.
2. **Trend fit.** OLS of `logit(rate)` on `x` over the distinct snapshots.
   Slopes are shrunk toward the cross-trophy median slope (hierarchical
   shrinkage), and the trend median is blended with persistence (the last
   rate) by a weight growing with the number of distinct snapshots. Slope
   and total median move are clipped.
3. **Flat history? Anticipate the refresh.** When the history never changed
   at all, no trend exists — the page is stale and a refresh is overdue.
   A refresh dilutes every rate except 100% (newly tracked players hold
   almost nothing yet; the rules describe the first refresh taking players
   tracked 638 → 3,868 while every rate but the 100% one fell). The median
   is shifted down by an amount growing with the square root of the stale
   hours and decaying with horizon, and the spread is widened. This branch
   only fires on a completely flat history.
4. **Honest spread.** Random-walk term `0.5·√days` (the house baseline's),
   plus the trend fit's residual SD, plus the robust SD of logit jumps
   between distinct snapshots (refresh volatility), combined in quadrature,
   floored at 0.30 and capped at 2.50 logit.

An earlier version also averaged in a counts model (separate OLS fits of
log earners and log players tracked). Backtests showed it explodes for
high-rate trophies — the two linear extrapolations diverge and their ratio
exceeds 100% — so it was removed; the logit-rate fit already embodies
earners-over-players dynamics.

## Validation (local; the kit has no practice answers — grading is server-only)

- **Practice p00/p01 vs reconstructed answers.** The rules state the
  04:30Z refresh (players 638 → 3,868, trophies earned 943 → 4,276) and
  earners per trophy are exact integers, so the answer is determined
  except for how the +103 new earners spread over the 54 non-100%
  trophies. Scored under 4 distribution scenarios (proportional, uniform,
  sqrt-weighted, top-5-heavy):
  - **This entry: 0.18–0.27** mean pinball loss, 90% coverage 0.98
  - House persistence baseline: 1.39–1.58, coverage 0.00 (rules say ~1.51)
- **Synthetic final-like backtest** (2 days of history with real growth +
  refresh steps every ~3h, target +4.5 days, 3 seeds): this entry
  0.24, house trend baseline 0.22, house persistence 0.54–0.57. On
  saturating (bending) truth this entry beats the trend baseline
  (0.24 vs 0.25–0.28); on exactly log-linear truth it trails it
  slightly (0.24 vs 0.22) — the shrinkage/blend is insurance against
  saturation, which the real week-one curves will show.
- **Edge cases** (all produce valid output, no crash): 1-read history,
  0% / 100% trophies, null rates, empty history (tier priors), corrupt
  JSON (skipped, other cases still written), 200,000-minute horizon.
- Deterministic: two runs byte-identical. Runtime ~0.04 s/case
  (limit 8 s).

## Files

- `main.py` — the forecaster
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`)
- `.makefun/owner.json` — payout wallet
