"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy, five quantiles of its Exophase earn rate
(percent of tracked players holding it) at the case's target, scored by
mean pinball loss on the logit scale.

Model (per trophy, all in logit space, x = ln(1 + hours since release)):
  1. Collapse the history to *distinct* snapshots: Exophase refreshes a
     game's counts on its own schedule, hours apart, so consecutive
     identical reads carry no information and would overweight stale
     periods in any fit.
  2. When at least two distinct snapshots exist, fit the rate curve
     directly: OLS of logit(rate) on x.  Slopes are shrunk toward the
     cross-trophy median slope (hierarchical shrinkage: one trophy's
     noisy slope borrows strength from all 55), and the trend median
     is blended with persistence (the last rate) by a weight growing
     with the number of distinct snapshots.  Slope and total median
     move are clipped, so a wild fit can never explode.  (An earlier
     version also averaged in a counts model -- separate OLS fits of
     log earners and log players tracked -- but backtests showed it
     explodes for high-rate trophies: the two linear extrapolations
     diverge and their ratio exceeds 100%.  The logit-rate fit
     already embodies earners-over-players dynamics, so the counts
     model was removed.)
  3. When the history never changed at all (a single distinct
     snapshot), no trend exists.  The page is then stale, and a
     refresh is overdue: a refresh dilutes every rate except 100%,
     because the newly tracked players hold almost nothing yet
     (the rules describe the first refresh taking players tracked
     638 -> 3,868 while every rate but the 100% one fell).  The
     median is shifted down by S * (1 - r/100) logit, S growing with
     the square root of the stale hours and decaying with horizon,
     and the spread is widened.  This branch only fires on a
     completely flat history, so it cannot touch a normal final case
     with two days of real reads.
  4. Spread: random-walk term 0.5*sqrt(days) (the house baseline's),
     plus the trend fit's residual SD, plus the robust SD of the
     logit jumps between distinct snapshots (refresh volatility),
     combined in quadrature, floored and capped.  Wider when the
     fit is based on few points.

Stdlib only, deterministic, no network, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- tuning constants (chosen by backtest, see PROGRESS.md) ---
RW_SIGMA = 0.5          # logit random-walk sigma per sqrt(day)
MIN_SIGMA = 0.30
MAX_SIGMA = 2.50
FLAT_SPREAD = 0.75      # extra logit sigma when history never changed
STALE_K = 0.75          # median down-shift per sqrt(stale hour), flat branch
STALE_SHIFT_CAP = 1.50
MAX_MEDIAN_MOVE = 4.0   # cap on |median logit - last-rate logit|
SLOPE_CLIP = (-2.0, 4.0)
TIER_PRIOR = {          # fallback median % when a trophy has no usable rate
    "platinum": 0.10, "gold": 0.30, "silver": 0.50, "bronze": 1.00,
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(value):
    """ISO-8601 string -> aware datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def parse_release_anchor(case, history):
    game = case.get("game") or {}
    raw = game.get("release_date")
    dt = None
    if isinstance(raw, str) and raw:
        dt = parse_time(raw)
        if dt is None:
            try:
                dt = datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                dt = None
    if dt is None:
        for read in history:
            dt = parse_time(read.get("at")) or parse_time(read.get("slot"))
            if dt is not None:
                break
    return dt


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    return (a, b, sd)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def robust_sd(values):
    """Median-absolute-deviation based SD; 0.0 when too few values."""
    if len(values) < 2:
        return 0.0
    med = median(values)
    mad = median([abs(v - med) for v in values])
    return 1.4826 * mad


def distinct_snapshots(history, n_trophies):
    """Collapse consecutive identical reads (same players + same rates)."""
    snaps = []
    prev_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        rates = list(rates[:n_trophies]) + [None] * max(0, n_trophies - len(rates))
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append({
            "at": parse_time(read.get("at")) or parse_time(read.get("slot")),
            "players": read.get("players_tracked"),
            "rates": rates,
        })
    return snaps


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    try:
        horizon_min = float(case.get("horizon_minutes") or 0.0)
    except (TypeError, ValueError):
        horizon_min = 0.0
    if horizon_min <= 0:
        at = parse_time(case.get("at"))
        tgt = parse_time(case.get("target"))
        if at is not None and tgt is not None:
            horizon_min = max(0.0, (tgt - at).total_seconds() / 60.0)
    days = max(horizon_min, 0.0) / 1440.0

    anchor = parse_release_anchor(case, history)
    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))

    snaps = distinct_snapshots(history, len(trophies))
    n_distinct = len(snaps)

    # Time coordinate of each distinct snapshot and of the target.
    def x_of(dt):
        if dt is None or anchor is None:
            return 0.0
        hours = (dt - anchor).total_seconds() / 3600.0
        return math.log1p(max(hours, 0.0))

    xs = [x_of(s["at"]) for s in snaps]
    x_target = x_of(target)
    if target is not None and checkpoint is not None and anchor is not None:
        # Target from the case beats any clock skew in the reads.
        hours_t = (target - anchor).total_seconds() / 3600.0
        x_target = math.log1p(max(hours_t, 0.0))
    x_last = xs[-1] if xs else 0.0

    # --- refresh-jump volatility: robust SD of per-trophy logit steps
    # between consecutive distinct snapshots (all trophies pooled). ---
    steps = []
    for prev, cur in zip(snaps, snaps[1:]):
        for i in range(len(trophies)):
            r0, r1 = prev["rates"][i], cur["rates"][i]
            if r0 is None or r1 is None:
                continue
            if r0 <= 0 or r0 >= 100 or r1 <= 0 or r1 >= 100:
                continue
            steps.append(logit(r1) - logit(r0))
    jump_sd = robust_sd(steps) if len(steps) >= 4 else 0.35

    # --- first pass: per-trophy OLS slopes, for the median slope. ---
    raw_slopes = []
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts = []
        for s, x in zip(snaps, xs):
            r = s["rates"][i]
            if r is None or r <= 0 or r >= 100:
                continue
            pts.append((x, logit(r)))
        fit = ols([p[0] for p in pts], [p[1] for p in pts]) \
            if len(pts) >= 3 and (max(p[0] for p in pts) - min(p[0] for p in pts)) > 0.05 else None
        slope = None
        if fit is not None:
            slope = min(max(fit[1], SLOPE_CLIP[0]), SLOPE_CLIP[1])
            raw_slopes.append(slope)
        per_trophy.append({"pts": pts, "fit": fit, "slope": slope})
    med_slope = median(raw_slopes) if raw_slopes else 0.0

    # --- staleness for the flat-history branch. ---
    stale_hours = 0.0
    if snaps and checkpoint is not None and snaps[0]["at"] is not None:
        stale_hours = max(0.0, (checkpoint - snaps[0]["at"]).total_seconds() / 3600.0)
    flat_shift = min(STALE_SHIFT_CAP,
                     STALE_K * math.sqrt(max(stale_hours, 0.0))) * math.exp(-days / 2.0)

    zvals = [norm_ppf(float(t)) for t in levels]
    rows = []
    for i, trophy in enumerate(trophies):
        # Last usable rate for this trophy (fall back: any read, then tier prior).
        last_rate = None
        last_players = None
        for s in reversed(snaps):
            if s["rates"][i] is not None:
                last_rate = s["rates"][i]
                last_players = s["players"]
                break
        if last_rate is None:
            for read in reversed(history):
                rates = read.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    last_rate = rates[i]
                    last_players = read.get("players_tracked")
                    break
        if last_rate is None or not isinstance(last_rate, (int, float)) \
                or not math.isfinite(last_rate):
            last_rate = TIER_PRIOR.get(str(trophy.get("tier", "")).lower(), 1.0)
        last_rate = min(max(float(last_rate), 0.0), 100.0)
        base_logit = logit(last_rate)

        info = per_trophy[i]
        if n_distinct >= 2 and info["fit"] is not None and len(info["pts"]) >= 3:
            # (a) logit-rate trend, slope shrunk to the median slope.
            a, b, resid = info["fit"]
            w = n_distinct / (n_distinct + 4.0)
            b_shrunk = w * info["slope"] + (1.0 - w) * med_slope
            b_shrunk = min(max(b_shrunk, SLOPE_CLIP[0]), SLOPE_CLIP[1])
            a_shrunk = sum(y - b_shrunk * x for x, y in info["pts"]) / len(info["pts"])
            m_line = a_shrunk + b_shrunk * x_target
            # Blend trend with persistence; more distinct reads -> more trend.
            w2 = n_distinct / (n_distinct + 2.0)
            med = (1.0 - w2) * base_logit + w2 * m_line
            resid_term = max(resid, 0.05)
            sigma = math.sqrt((RW_SIGMA * math.sqrt(days)) ** 2
                              + (1.3 * resid_term) ** 2
                              + (0.6 * jump_sd) ** 2
                              + (0.3 / math.sqrt(n_distinct)) ** 2
                              + 0.10 ** 2)
        elif n_distinct == 2 and len(info["pts"]) >= 2:
            # Two distinct snapshots: a weak trend, heavily shrunk.
            (x0, y0), (x1, y1) = info["pts"][0], info["pts"][-1]
            if x1 - x0 > 0.05:
                b_raw = min(max((y1 - y0) / (x1 - x0), SLOPE_CLIP[0]), SLOPE_CLIP[1])
                b_shrunk = 0.25 * b_raw + 0.75 * med_slope
                m_trend = y1 + b_shrunk * (x_target - x1)
                med = 0.5 * base_logit + 0.5 * m_trend
            else:
                med = base_logit
            sigma = math.sqrt((RW_SIGMA * math.sqrt(days)) ** 2
                              + (0.6 * jump_sd) ** 2 + 0.45 ** 2)
        else:
            # Flat history (0-1 distinct snapshots): no trend to fit.
            # Anticipate the overdue refresh diluting the rate down,
            # except for a trophy already at 100%.
            med = base_logit - flat_shift * (1.0 - last_rate / 100.0)
            sigma = math.sqrt((RW_SIGMA * math.sqrt(days)) ** 2 + FLAT_SPREAD ** 2)

        med = min(max(med, base_logit - MAX_MEDIAN_MOVE),
                  base_logit + MAX_MEDIAN_MOVE)
        sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)
        qs = []
        prev_q = None
        for z in zvals:
            q = round4(min(max(inv_logit(med + z * sigma), 0.0), 100.0))
            if prev_q is not None and q < prev_q:
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def forecast_case_safe(case):
    """Never raise: fall back to a flat forecast with a wide spread."""
    try:
        return forecast_case(case)
    except Exception:
        trophies = []
        try:
            trophies = (case or {}).get("trophies") or []
        except Exception:
            trophies = []
        rows = []
        for trophy in trophies:
            try:
                tid = trophy["id"]
            except Exception:
                continue
            rows.append({"id": tid, "q": [0.05, 0.5, 1.0, 2.0, 5.0]})
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        paths = sorted(INPUT_DIR.glob("*.in"))
    except Exception:
        paths = []
    for path in paths:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
        except Exception:
            continue
        forecast = forecast_case_safe(case)
        try:
            (OUTPUT_DIR / (path.stem + ".out")).write_text(
                json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        except Exception:
            continue


if __name__ == "__main__":
    main()
