"""
Alex Sarr: model projects PTS, TRB, AST, STL, BLK for next 5 Wizards games.
Uses rolling individual stats + opponent defensive rank, win streak, size/matchup,
wingspan advantage, environment (home/away), and availability.
Each projected stat includes an explanation string and optional confidence band.
v1.0 – Baseline contextual projection model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

__version__ = "1.0.0"

# --- Rolling baseline ---
STAT_COLS = ["PTS", "TRB", "AST", "STL", "BLK"]
DEFAULT_WINDOW = 10
N_TEAMS = 30
BACKTEST_N_GAMES = 25  # 15-25 past games for validation


def rolling_projection(df: pd.DataFrame, stat_col: str, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Rolling mean for a stat (min_periods=1)."""
    return df[stat_col].rolling(window=window, min_periods=1).mean()


def rolling_std(df: pd.DataFrame, stat_col: str, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Rolling std for a stat."""
    return df[stat_col].rolling(window=window, min_periods=1).std()


def prob_over_threshold(
    roll_mean: pd.Series, roll_std: pd.Series, threshold: float
) -> np.ndarray:
    """P(stat > threshold) under normal approximation. Handles zero/NaN std."""
    prob_over = np.where(
        roll_std > 0,
        1.0 - norm.cdf((threshold - roll_mean) / roll_std),
        np.where(
            roll_std == 0,
            np.where(roll_mean > threshold, 1.0, np.where(roll_mean < threshold, 0.0, 0.5)),
            np.nan,
        ),
    )
    return prob_over


def add_rolling_and_probs(
    df: pd.DataFrame,
    stat_columns: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
    pts_over: float = 25.0,
) -> pd.DataFrame:
    """Add rolling mean, rolling std, prob_over_25 for PTS; rolling means for TRB, AST, STL, BLK; and MP."""
    df = df.copy()
    stat_columns = stat_columns or STAT_COLS
    for col in stat_columns:
        if col not in df.columns:
            continue
        df[f"{col}_roll_mean"] = rolling_projection(df, col, window)
        df[f"{col}_roll_std"] = rolling_std(df, col, window)
    if "MP" in df.columns:
        df["MP_roll_mean"] = rolling_projection(df, "MP", window)
    if "PTS" in df.columns:
        rm = df["PTS_roll_mean"]
        rs = df["PTS_roll_std"]
        df["prob_over_25"] = prob_over_threshold(rm, rs, pts_over)
    return df


def last_rolling_values(df: pd.DataFrame, stat_columns: list[str] | None = None) -> dict[str, float]:
    """Get latest rolling mean (and std for PTS) from the end of the game log."""
    stat_columns = stat_columns or STAT_COLS
    out = {}
    for col in stat_columns:
        mean_col = f"{col}_roll_mean"
        if mean_col in df.columns:
            out[col] = float(df[mean_col].iloc[-1])
        elif col in df.columns:
            out[col] = float(df[col].iloc[-1])
    if "PTS_roll_std" in df.columns:
        out["PTS_roll_std"] = float(df["PTS_roll_std"].iloc[-1])
    if "prob_over_25" in df.columns:
        out["prob_over_25"] = float(df["prob_over_25"].iloc[-1])
    if "MP_roll_mean" in df.columns:
        out["MP"] = float(df["MP_roll_mean"].iloc[-1])
    elif "MP" in df.columns:
        out["MP"] = float(df["MP"].iloc[-1])
    return out


def last_rolling_stds(df: pd.DataFrame, stat_columns: list[str] | None = None) -> dict[str, float]:
    """Rolling std at end of game log for confidence bands when MAE not available."""
    stat_columns = stat_columns or STAT_COLS
    out = {}
    for col in stat_columns:
        std_col = f"{col}_roll_std"
        if std_col in df.columns:
            out[col] = float(df[std_col].iloc[-1])
    return out


def rolling_values_before_row(
    df: pd.DataFrame,
    row_idx: int,
    window: int = DEFAULT_WINDOW,
    stat_columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Rolling means using only rows [0, row_idx-1] (exclude current game).
    Used for backtest: "projection for game at row_idx" = baseline from prior games.
    """
    stat_columns = stat_columns or STAT_COLS
    if row_idx <= 0:
        return {c: 0.0 for c in stat_columns}
    prior = df.iloc[:row_idx]
    out = {}
    for col in stat_columns:
        if col not in prior.columns:
            continue
        s = prior[col].rolling(window=window, min_periods=1).mean()
        out[col] = float(s.iloc[-1])
    if "MP" in prior.columns:
        s = prior["MP"].rolling(window=window, min_periods=1).mean()
        out["MP"] = float(s.iloc[-1])
    return out


# --- Opponent / matchup factors (0–1 scale: higher = better for Sarr) ---
def opponent_strength_factor(defensive_rating_rank: int | None, n_teams: int = 30) -> float:
    """Convert defensive rank (1=best defense) to factor. Higher = weaker defense = better for Sarr."""
    if defensive_rating_rank is None:
        return 0.5
    # Rank 1 = best D = hardest; Rank 30 = worst D = easiest
    return (defensive_rating_rank - 1) / (n_teams - 1) if n_teams > 1 else 0.5


def availability_factor(opponent_missing_key_players: bool) -> float:
    """1.0 = they're at full strength (harder), 0.0 = key players out (easier for Sarr)."""
    return 0.0 if opponent_missing_key_players else 1.0


def wingspan_factor(sarr_advantage: float) -> float:
    """Map wingspan advantage to 0–1. positive = Sarr has advantage. Clamp to [0,1]."""
    # e.g. Sarr ~7'4"; if opponent primary defender avg is 7'0", advantage ~+4 inches -> boost
    # Simple: 0 = even, 0.5 = small adv, 1 = big adv. Input in inches or normalized.
    return float(np.clip(0.5 + sarr_advantage * 0.1, 0.0, 1.0))


def breakout_score(
    prob_over_25: float,
    opp_strength: float,
    availability: float,
    wingspan: float,
    w_prob: float = 0.4,
    w_opp: float = 0.25,
    w_avail: float = 0.2,
    w_wing: float = 0.15,
) -> float:
    """
    Composite 0–1 score for "due for a breakout" (higher = more likely).
    """
    missing_helps = 1.0 - availability
    return float(
        w_prob * (prob_over_25 if not np.isnan(prob_over_25) else 0.5)
        + w_opp * opp_strength
        + w_avail * missing_helps
        + w_wing * wingspan
    )


# --- Model projections: defensive rating is central; model prides itself on TRB & BLK ---
# PTS is noisier than TRB; BLK is highly volatile. Opponent defensive rank drives big shifts.
TOP10_DEF_RANK = 10  # "good team" = top-10 defense
EXTRA_PENALTY_VS_TOP10 = 0.04  # extra cut vs good teams so we don't overestimate


def _defensive_adjustment(rank: int | None, stat: str) -> float:
    """Opponent defensive rank (1=best, 30=worst) drives projection shifts. Stronger penalty vs good teams."""
    if rank is None:
        return 0.0
    scales = {"PTS": 0.14, "AST": 0.10, "TRB": 0.08, "STL": 0.08, "BLK": 0.06}
    s = scales.get(stat, 0.08)
    adj = ((rank - (N_TEAMS + 1) / 2) / N_TEAMS) * s
    if rank <= TOP10_DEF_RANK:
        adj -= EXTRA_PENALTY_VS_TOP10 if stat == "PTS" else (EXTRA_PENALTY_VS_TOP10 * 0.5)
    return adj


def _win_streak_adjustment(win_streak: int) -> float:
    """Hot opponent = slightly tougher; small negative adjustment. Cap at 5."""
    if win_streak <= 0:
        return 0.0
    return -0.01 * min(int(win_streak), 5)


def _wingspan_adjustment(inches: float, stat: str) -> float:
    """Sarr's wingspan advantage in inches -> boost. BLK/TRB most sensitive."""
    if inches <= 0:
        return 0.0
    per_inch = {"BLK": 0.025, "TRB": 0.02, "PTS": 0.008, "STL": 0.01, "AST": 0.005}
    return min(inches * per_inch.get(stat, 0.01), 0.15)


def _environment_adjustment(is_home: bool) -> float:
    """Home court boost."""
    return 0.02 if is_home else 0.0


def _availability_adjustment(key_players_out: bool, stat: str) -> float:
    """Key defenders/centers out -> boost; larger for scoring stats."""
    if not key_players_out:
        return 0.0
    return 0.04 if stat in ("PTS", "TRB", "BLK") else 0.02


def _matchup_adjustment(stat: str, game: dict) -> float:
    """Small penalty for PTS/BLK when rim protector present; optional switch/zone."""
    adj = 0.0
    if game.get("rim_protector_present"):
        if stat == "PTS":
            adj -= 0.03
        elif stat == "BLK":
            adj -= 0.04
    if game.get("switch_heavy_defense"):
        if stat == "AST":
            adj -= 0.02
    if game.get("zone_frequency") == "high":
        adj -= 0.02
    elif game.get("zone_frequency") == "medium":
        adj -= 0.01
    return adj


# --- Explanation layer: human-readable reason strings per factor ---
def _defensive_reason(rank: int | None) -> str | None:
    if rank is None:
        return None
    if rank >= 22:
        return "+ weak defense"
    if rank <= TOP10_DEF_RANK:
        return "− strong defense (top-10)"
    return None


def _win_streak_reason(win_streak: int) -> str | None:
    if win_streak <= 0:
        return None
    return "− opponent win streak"


def _wingspan_reason(inches: float) -> str | None:
    if inches <= 0:
        return None
    return "+ wingspan advantage"


def _environment_reason(is_home: bool) -> str | None:
    if is_home:
        return "+ home"
    return "− away"


def _availability_reason(key_players_out: bool) -> str | None:
    if not key_players_out:
        return None
    return "+ key players out"


def _matchup_reasons(stat: str, game: dict) -> list[str]:
    reasons = []
    if game.get("rim_protector_present") and stat in ("PTS", "BLK"):
        reasons.append("− rim protector")
    if game.get("switch_heavy_defense") and stat == "AST":
        reasons.append("− switch-heavy D")
    z = game.get("zone_frequency")
    if z == "high":
        reasons.append("− zone")
    elif z == "medium":
        reasons.append("− some zone")
    return reasons


def explain_stat(
    stat: str,
    *,
    opp_def_rank: int | None = None,
    opponent_win_streak: int = 0,
    wingspan_advantage_inches: float = 0.0,
    is_home: bool = False,
    key_players_out: bool = False,
    game: dict | None = None,
) -> str:
    """
    Build a short reason string for the projection (interpretable model).
    Example: "+ weak defense, + home, − opponent win streak"
    """
    parts = []
    r = _defensive_reason(opp_def_rank)
    if r:
        parts.append(r)
    r = _win_streak_reason(opponent_win_streak)
    if r:
        parts.append(r)
    r = _wingspan_reason(wingspan_advantage_inches)
    if r:
        parts.append(r)
    r = _environment_reason(is_home)
    if r:
        parts.append(r)
    r = _availability_reason(key_players_out)
    if r:
        parts.append(r)
    if game:
        parts.extend(_matchup_reasons(stat, game))
    if not parts:
        return "baseline (neutral matchup)"
    return ", ".join(parts)


def explain_minutes(game: dict) -> str:
    """Reason string for projected minutes."""
    parts = []
    is_home = (game.get("venue") or "").strip().lower() == "home"
    if is_home:
        parts.append("+ home")
    if game.get("opponent_missing_key_players"):
        parts.append("+ key players out")
    if not parts:
        return "baseline minutes"
    return ", ".join(parts)


def project_minutes(baseline_mp: float, game: dict) -> float:
    """Project minutes from baseline MP + environment and availability."""
    if not baseline_mp or baseline_mp <= 0:
        return 28.0
    is_home = (game.get("venue") or "").strip().lower() == "home"
    key_out = game.get("opponent_missing_key_players", False)
    mult = 1.0 + _environment_adjustment(is_home) + (0.03 if key_out else 0.0)
    return round(max(0, baseline_mp * mult), 1)


def project_stat(
    baseline: float,
    stat: str,
    *,
    opp_def_rank: int | None = None,
    opponent_win_streak: int = 0,
    wingspan_advantage_inches: float = 0.0,
    is_home: bool = False,
    key_players_out: bool = False,
    game: dict | None = None,
    baseline_minutes: float | None = None,
    proj_minutes: float | None = None,
) -> float:
    """
    Project one stat. If baseline_minutes and proj_minutes are set, use per-minute rate:
      proj = proj_minutes * (baseline / baseline_minutes) * (1 + adjustments).
    Otherwise: proj = baseline * (1 + adjustments).
    """
    mult = 1.0
    mult += _defensive_adjustment(opp_def_rank, stat)
    mult += _win_streak_adjustment(opponent_win_streak)
    mult += _wingspan_adjustment(wingspan_advantage_inches, stat)
    mult += _environment_adjustment(is_home)
    mult += _availability_adjustment(key_players_out, stat)
    if game:
        mult += _matchup_adjustment(stat, game)
    if baseline_minutes and baseline_minutes > 0 and proj_minutes is not None:
        rate = baseline / baseline_minutes
        proj = proj_minutes * rate * mult
    else:
        proj = baseline * mult
    proj = max(0.0, proj)
    return round(proj, 1)


def project_game(
    baseline: dict[str, float],
    game: dict,
) -> dict[str, float]:
    """
    Project minutes first, then each stat from per-minute rate * proj_minutes * adjustments.
    Returns dict with proj_MP (if MP baseline exists), proj_PTS, proj_TRB, ..., and reason_PTS, reason_TRB, ...
    """
    opp_rank = game.get("opponent_defensive_rank")
    win_streak = game.get("opponent_win_streak", 0) or 0
    wing_in = float(game.get("wingspan_advantage_inches") or 0)
    is_home = (game.get("venue") or "").strip().lower() == "home"
    key_out = game.get("opponent_missing_key_players", False)
    baseline_mp = baseline.get("MP") or 0.0
    proj_mp = project_minutes(baseline_mp, game) if baseline_mp > 0 else None
    out = {}
    if proj_mp is not None:
        out["MP"] = proj_mp
        out["reason_MP"] = explain_minutes(game)
    for stat in STAT_COLS:
        out[stat] = project_stat(
            baseline.get(stat, 0.0),
            stat,
            opp_def_rank=opp_rank,
            opponent_win_streak=win_streak,
            wingspan_advantage_inches=wing_in,
            is_home=is_home,
            key_players_out=key_out,
            game=game,
            baseline_minutes=baseline_mp if baseline_mp > 0 else None,
            proj_minutes=proj_mp,
        )
        out[f"reason_{stat}"] = explain_stat(
            stat,
            opp_def_rank=opp_rank,
            opponent_win_streak=win_streak,
            wingspan_advantage_inches=wing_in,
            is_home=is_home,
            key_players_out=key_out,
            game=game,
        )
    return out


def load_config(config_path: str | Path) -> dict:
    """Load next 5 games and opponent/matchup info from JSON."""
    path = Path(config_path)
    if not path.exists():
        return default_config()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_config() -> dict:
    """Default structure for next 5 games. Model uses no external lines."""
    return {
        "next_5_games": [
            {
                "date": "2025-02-08",
                "opponent": "BOS",
                "venue": "away",
                "opponent_defensive_rank": 3,
                "opponent_win_streak": 0,
                "opponent_missing_key_players": False,
                "wingspan_advantage_inches": 0.0,
                "rim_protector_present": True,
                "switch_heavy_defense": False,
                "zone_frequency": "low",
            },
            {
                "date": "2025-02-10",
                "opponent": "MIA",
                "venue": "home",
                "opponent_defensive_rank": 12,
                "opponent_win_streak": 2,
                "opponent_missing_key_players": True,
                "wingspan_advantage_inches": 2.0,
                "rim_protector_present": False,
                "switch_heavy_defense": False,
                "zone_frequency": "low",
            },
        ],
        "notes": "Optional matchup: rim_protector_present, switch_heavy_defense, zone_frequency (low/medium/high).",
    }


def _band_width(stat: str, mae_by_stat: dict | None, roll_stds: dict | None) -> float:
    """Band width for confidence interval: prefer MAE from backtest, else ±1 rolling std."""
    if mae_by_stat and stat in mae_by_stat:
        return mae_by_stat[stat]
    if roll_stds and stat in roll_stds and pd.notna(roll_stds[stat]) and roll_stds[stat] > 0:
        return float(roll_stds[stat])
    return 2.0  # default fallback


def run_analysis(
    df: pd.DataFrame,
    config: dict,
    window: int = DEFAULT_WINDOW,
    pts_over: float = 25.0,
    mae_by_stat: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Project PTS/TRB/AST/STL/BLK for next 5 games. Optionally add confidence bands (proj_*_lo, proj_*_hi)
    using MAE from backtest or rolling std.
    """
    df = add_rolling_and_probs(df, window=window, pts_over=pts_over)
    baseline = last_rolling_values(df)
    roll_stds = last_rolling_stds(df)

    games = config.get("next_5_games", [])
    rows = []
    for g in games[:5]:
        opp_rank = g.get("opponent_defensive_rank")
        missing = g.get("opponent_missing_key_players", False)
        wing_adv = float(g.get("wingspan_advantage_inches") or 0)

        opp_str = opponent_strength_factor(opp_rank)
        avail = availability_factor(missing)
        wing = wingspan_factor(wing_adv)
        prob_25 = baseline.get("prob_over_25", 0.5)
        if np.isnan(prob_25):
            prob_25 = 0.5
        score = breakout_score(prob_25, opp_str, avail, wing)

        proj = project_game(baseline, g)

        row = {
            "date": g.get("date", ""),
            "opponent": g.get("opponent", ""),
            "venue": g.get("venue", ""),
            "breakout_score": round(score, 3),
            "prob_over_25_pts": round(prob_25, 3),
            "opp_def_rank": opp_rank,
            "opponent_win_streak": g.get("opponent_win_streak", 0),
            "key_players_out": missing,
            "wingspan_adv_in": wing_adv,
        }
        if "MP" in proj:
            row["proj_MP"] = proj["MP"]
            row["reason_MP"] = proj.get("reason_MP", "")
        for stat in STAT_COLS:
            p = proj[stat]
            row[f"proj_{stat}"] = p
            row[f"reason_{stat}"] = proj.get(f"reason_{stat}", "")
            band = _band_width(stat, mae_by_stat, roll_stds)
            row[f"proj_{stat}_lo"] = round(max(0, p - band), 1)
            row[f"proj_{stat}_hi"] = round(p + band, 1)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(analysis_df: pd.DataFrame, baseline: dict) -> None:
    """Pretty-print baseline (rolling) and next 5 games with model projections."""
    print("\n--- Sarr baseline (rolling 10-game means) ---")
    if baseline.get("MP") is not None:
        print(f"  MP: {baseline['MP']:.2f}")
    for k in STAT_COLS:
        v = baseline.get(k)
        if v is not None:
            print(f"  {k}: {v:.2f}")
    if "prob_over_25" in baseline:
        print(f"  prob_over_25: {baseline['prob_over_25']:.2f}")

    print("\n--- Next 5 games: model projections with reasons (interpretable) ---")
    for _, r in analysis_df.iterrows():
        print(f"\n  {r['date']} vs {r['opponent']} ({r['venue']})")
        print(f"  Breakout score: {r['breakout_score']:.2f}  |  P(over 25 PTS): {r['prob_over_25_pts']:.2f}")
        print(f"  Opp def rank: {r['opp_def_rank']}  |  Win streak: {r.get('opponent_win_streak', 0)}  |  Key out: {r['key_players_out']}  |  Wingspan adv: {r['wingspan_adv_in']}\"")
        if "proj_MP" in r and pd.notna(r.get("proj_MP")):
            print(f"  MP: {r['proj_MP']}")
            print(f"    Reason: {r.get('reason_MP', '')}")
        for stat in STAT_COLS:
            val = r.get(f"proj_{stat}", "")
            lo = r.get(f"proj_{stat}_lo", "")
            hi = r.get(f"proj_{stat}_hi", "")
            reason = r.get(f"reason_{stat}", "")
            print(f"  {stat}: {val}  range: ({lo}, {hi})")
            print(f"    Reason: {reason}")


# --- STEP 1: Backtest validation (model projection vs actual) ---
def _parse_game_log_context(df: pd.DataFrame, row_idx: int) -> dict:
    """Extract date, opp, is_home, back_to_back from game log row for segmentation."""
    out = {}
    row = df.iloc[row_idx]
    if "Date" in df.columns:
        out["game_date"] = row.get("Date", "")
    opp = row.get("Opp", row.get("OPP", ""))
    if opp is not None and str(opp).strip():
        out["opponent"] = str(opp).strip()
        out["is_home"] = not str(opp).strip().startswith("@")
    if "Date" in df.columns and row_idx > 0:
        try:
            d = pd.to_datetime(row.get("Date"), errors="coerce")
            d_prev = pd.to_datetime(df.iloc[row_idx - 1].get("Date"), errors="coerce")
            if pd.notna(d) and pd.notna(d_prev):
                out["back_to_back"] = (d - d_prev).days <= 1
            else:
                out["back_to_back"] = False
        except Exception:
            out["back_to_back"] = False
    return out


def backtest_validation(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    n_games: int = BACKTEST_N_GAMES,
    add_segment_columns: bool = True,
) -> pd.DataFrame:
    """
    Pretend the last n_games (15-25) were "upcoming". Use only info available before each game.
    Returns DataFrame with actuals, proj_*, *_error, and optional game_date, opponent, is_home, back_to_back.
    """
    df = df.copy()
    if "MP" not in df.columns:
        df["MP"] = 28.0
    n = len(df)
    if n < window + 1:
        return pd.DataFrame()
    start_idx = max(window, n - n_games)
    rows = []
    for i in range(start_idx, n):
        baseline = rolling_values_before_row(df, i, window=window)
        row = {"row_idx": i}
        if add_segment_columns:
            row.update(_parse_game_log_context(df, i))
        for stat in STAT_COLS:
            actual_val = df.iloc[i][stat] if stat in df.columns else np.nan
            proj_val = baseline.get(stat, 0.0)
            row[stat] = actual_val
            row[f"proj_{stat}"] = round(proj_val, 1)
            row[f"{stat}_error"] = (actual_val - proj_val) if pd.notna(actual_val) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_backtest(backtest_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute MAE (via sklearn) and bias per stat. Returns MAE for confidence bands.
    """
    if backtest_df.empty:
        return {}
    out = {}
    for stat in STAT_COLS:
        actual_col = stat
        proj_col = f"proj_{stat}"
        if actual_col not in backtest_df.columns or proj_col not in backtest_df.columns:
            continue
        y_true = backtest_df[actual_col].dropna()
        y_pred = backtest_df.loc[y_true.index, proj_col]
        if len(y_true) == 0:
            continue
        mae = mean_absolute_error(y_true, y_pred)
        err = backtest_df[f"{stat}_error"].dropna()
        bias = float(err.mean()) if len(err) else 0.0
        out[stat] = {"MAE": float(mae), "bias": bias}
    return out


def get_mae_by_stat(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Extract MAE per stat for confidence bands."""
    return {stat: m["MAE"] for stat, m in (metrics or {}).items()}


def compute_noise_volatility(
    backtest_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Per-stat: mean actual (scale), normalized MAE (MAE/mean), std of error (volatility).
    Answers: Is PTS noisier than TRB? Are blocks wildly volatile?
    """
    out = {}
    for stat in STAT_COLS:
        if stat not in backtest_df.columns or stat not in metrics:
            continue
        actual = backtest_df[stat].dropna()
        if len(actual) < 2:
            continue
        err_col = f"{stat}_error"
        err = backtest_df.loc[actual.index, err_col].dropna()
        mean_actual = float(actual.mean())
        mae = metrics[stat]["MAE"]
        out[stat] = {
            "mean_actual": mean_actual,
            "normalized_mae": mae / mean_actual if mean_actual > 0 else 0,
            "std_error": float(err.std()) if len(err) > 1 else 0,
        }
    return out


def print_noise_volatility_report(
    backtest_df: pd.DataFrame,
    metrics: dict,
    noise: dict[str, dict[str, float]],
) -> None:
    """
    Answer: Is PTS noisier than TRB? Are blocks wildly volatile?
    Model prides itself on rebounds and blocks (TRB, BLK).
    """
    if not noise or backtest_df.empty:
        return
    print("\n--- Noise & volatility (model focus: TRB & BLK) ---")
    print("  Normalized MAE = MAE / mean(actual); higher = noisier.")
    print("  Std(error) = volatility of projection error.")
    for stat in ["TRB", "BLK", "PTS", "AST", "STL"]:
        if stat not in noise:
            continue
        n = noise[stat]
        nm = n.get("normalized_mae", 0)
        se = n.get("std_error", 0)
        print(f"  {stat}:  normalized MAE = {nm:.2f}  |  std(error) = {se:.2f}")
    pts_n = noise.get("PTS", {}).get("normalized_mae")
    trb_n = noise.get("TRB", {}).get("normalized_mae")
    blk_n = noise.get("BLK", {}).get("normalized_mae")
    if pts_n is not None and trb_n is not None:
        noisier = "PTS is noisier than TRB" if pts_n > trb_n else "TRB is noisier than PTS"
        print(f"  → {noisier} (normalized MAE: PTS {pts_n:.2f}, TRB {trb_n:.2f})")
    if blk_n is not None:
        others = [noise[s].get("normalized_mae") for s in STAT_COLS if s != "BLK" and s in noise]
        avg_other = np.mean(others) if others else 0
        if blk_n > avg_other * 1.2:
            print(f"  → Blocks are highly volatile (normalized MAE {blk_n:.2f} vs avg {avg_other:.2f})")
    print("  → Model prides itself on rebounds (TRB) and blocks (BLK); defensive rating drives big shifts.")


def print_defense_tier_summary(backtest_df: pd.DataFrame, metrics: dict) -> None:
    """
    Does the model overestimate vs good teams? Bias vs top-10 defense (when opp_def_rank available).
    """
    if "top10_defense" not in backtest_df.columns:
        print("\n--- Vs good teams (top-10 defense) ---")
        print("  Add historical_context (opp_def_rank) to backtest to check if model overestimates vs good teams.")
        return
    bt = backtest_df
    mask = bt["top10_defense"].fillna(False) == True
    if mask.sum() < 2:
        return
    print("\n--- Vs good teams (top-10 defense) – overestimate check ---")
    print("  Opponent defensive rating should play a big part; bias < 0 = model overprojects.")
    for stat in STAT_COLS:
        err_col = f"{stat}_error"
        if err_col not in bt.columns:
            continue
        err = bt.loc[mask, err_col].dropna()
        if len(err) < 2:
            continue
        bias = float(err.mean())
        over = "overestimates" if bias < 0 else "underestimates"
        print(f"  {stat} vs top-10 D (n={mask.sum()}): bias = {bias:+.2f}  →  model {over}")
    print("  (Supply historical_context with opp_def_rank for this segment.)")


def print_backtest_summary(backtest_df: pd.DataFrame, metrics: dict) -> None:
    """Backtest: games count; TRB & BLK first (model pride), then MAE/bias; noise and vs-good-teams."""
    if backtest_df.empty:
        print("Backtest: not enough games (need at least window+1).")
        return
    print("\n--- STEP 1: Model validation (backtest) ---")
    print(f"  Games in backtest: {len(backtest_df)}")
    print("  Error = actual - projected (positive = model underprojected)")
    order = ["TRB", "BLK", "PTS", "AST", "STL"]
    for stat in order:
        if stat not in metrics:
            continue
        m = metrics[stat]
        print(f"  {stat}:  MAE = ±{m['MAE']:.2f}  |  bias = {m['bias']:+.2f}")
    print("  (bias > 0: underproject; bias < 0: overproject)")
    noise = compute_noise_volatility(backtest_df, metrics)
    print_noise_volatility_report(backtest_df, metrics, noise)
    print_defense_tier_summary(backtest_df, metrics)


# --- STEP 3: Segment by game type (find flaws) ---
def segment_backtest(
    backtest_df: pd.DataFrame,
    game_log: pd.DataFrame,
    historical_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Enrich backtest with segment columns. historical_context can have row_idx or game_date + opponent
    and columns: opp_def_rank, rim_protector_present. Merged so we can segment by top-10/bottom-10 D and rim.
    """
    bt = backtest_df.copy()
    if historical_context is not None and not historical_context.empty:
        if "row_idx" in historical_context.columns:
            bt = bt.merge(historical_context, on="row_idx", how="left")
        elif "game_date" in bt.columns and "game_date" in historical_context.columns:
            bt = bt.merge(historical_context, on="game_date", how="left")
    if "opp_def_rank" in bt.columns:
        bt["top10_defense"] = bt["opp_def_rank"].le(10)
        bt["bottom10_defense"] = bt["opp_def_rank"].ge(N_TEAMS - 9)
    return bt


def segment_metrics(
    backtest_df: pd.DataFrame,
) -> list[dict]:
    """
    Break backtest results by segment. Returns list of {segment_name, segment_value, stat, MAE, bias, n}.
    """
    segments = []
    bt = backtest_df

    def _add(segment_name: str, mask: pd.Series, label: str):
        mask = mask.fillna(False)
        if mask.sum() < 2:
            return
        sub = bt.loc[mask]
        for stat in STAT_COLS:
            err_col = f"{stat}_error"
            if err_col not in sub.columns:
                continue
            err = sub[err_col].dropna()
            if len(err) < 2:
                continue
            segments.append({
                "segment": segment_name,
                "value": label,
                "stat": stat,
                "MAE": float(np.abs(err).mean()),
                "bias": float(err.mean()),
                "n": int(mask.sum()),
            })

    if "is_home" in bt.columns:
        _add("venue", bt["is_home"].fillna(False) == True, "home")
        _add("venue", bt["is_home"].fillna(False) == False, "away")
    if "back_to_back" in bt.columns:
        _add("back_to_back", bt["back_to_back"].fillna(False) == True, "yes")
        _add("back_to_back", bt["back_to_back"].fillna(False) == False, "no")
    if "top10_defense" in bt.columns:
        _add("defense_tier", bt["top10_defense"].fillna(False) == True, "top10_defense")
    if "bottom10_defense" in bt.columns:
        _add("defense_tier", bt["bottom10_defense"].fillna(False) == True, "bottom10_defense")
    if "rim_protector_present" in bt.columns:
        _add("rim_protector", bt["rim_protector_present"].fillna(False) == True, "yes")
        _add("rim_protector", bt["rim_protector_present"].fillna(False) == False, "no")
    return segments


def print_segment_report(backtest_df: pd.DataFrame) -> None:
    """Print MAE/bias by segment to surface flaws (e.g. overstates vs top-10 D with rim protector)."""
    segs = segment_metrics(backtest_df)
    if not segs:
        return
    print("\n--- STEP 3: Segment analysis (where are the flaws?) ---")
    from itertools import groupby
    for key, group in groupby(sorted(segs, key=lambda x: (x["segment"], x["value"])), key=lambda x: (x["segment"], x["value"])):
        seg_name, seg_val = key
        items = list(group)
        n = items[0]["n"] if items else 0
        print(f"  {seg_name} = {seg_val} (n={n})")
        for x in items:
            print(f"    {x['stat']}: MAE ±{x['MAE']:.2f}, bias {x['bias']:+.2f}")


# --- Game log loading: CSV, then optional fetch from Basketball-Reference ---
SARR_GAMELOG_URL = "https://www.basketball-reference.com/players/s/sarral01/gamelog/2025"


def load_game_log(
    folder: str | Path | None = None,
    csv_name: str = "sarr_game_log.csv",
    fetch_from_url: bool = True,
) -> pd.DataFrame | None:
    """
    Load Sarr game log. Tries (1) CSV in folder, then (2) fetch from Basketball-Reference URL.
    Returns None if both fail. Column names are normalized to PTS, MP, TRB, AST, STL, BLK.
    """
    folder = Path(folder or __file__).parent if "__file__" in dir() else Path(".")
    folder = Path(folder)
    csv_path = folder / csv_name

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif fetch_from_url:
        df = None
        try:
            tables = pd.read_html(SARR_GAMELOG_URL)
            for t in tables:
                if "PTS" in t.columns and len(t) >= 5:
                    df = t.copy()
                    break
        except Exception:
            pass
    else:
        df = None

    if df is None:
        return None

    # Normalize column names to PTS, MP, TRB, AST, STL, BLK
    for std in ["PTS", "MP", "TRB", "AST", "STL", "BLK"]:
        if std not in df.columns:
            for c in df.columns:
                if c.upper() == std:
                    df = df.rename(columns={c: std})
                    break
    # Drop non-game rows (header repeats, "Season" totals)
    for col in ["PTS", "TRB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PTS"], how="all")
    if "PTS" in df.columns:
        df = df[df["PTS"].notna()].reset_index(drop=True)
    return df


if __name__ == "__main__":
    config_path = Path(__file__).parent / "sarr_next5_config.json"
    config = load_config(config_path)

    df = load_game_log(fetch_from_url=True)
    if df is None or len(df) < 2:
        print("No game log found (no sarr_game_log.csv and URL fetch failed or empty). Using placeholder data.")
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "PTS": np.clip(np.cumsum(np.random.randn(n) * 2) + 12, 4, 35),
            "MP": np.random.randint(24, 36, n),
            "TRB": np.random.randint(4, 14, n),
            "AST": np.random.randint(1, 5, n),
            "STL": np.random.randint(0, 3, n),
            "BLK": np.random.randint(0, 4, n),
        })

    df = add_rolling_and_probs(df)
    baseline = last_rolling_values(df)

    # STEP 1: Backtest (15-25 past games; only info available before each game)
    backtest_df = backtest_validation(df, window=DEFAULT_WINDOW, n_games=BACKTEST_N_GAMES)
    mae_by_stat = None
    if not backtest_df.empty:
        metrics = evaluate_backtest(backtest_df)
        print_backtest_summary(backtest_df, metrics)
        mae_by_stat = get_mae_by_stat(metrics)
        backtest_path = Path(__file__).parent / "sarr_backtest_validation.csv"
        backtest_df.to_csv(backtest_path, index=False)
        print(f"  Backtest details saved to: {backtest_path}")
        # STEP 3: Segment by game type (home/away, B2B, etc.)
        print_segment_report(backtest_df)

    # STEP 2: Projections with confidence bands (proj_lo, proj_hi from MAE or rolling std)
    analysis_df = run_analysis(df, config, mae_by_stat=mae_by_stat)
    print_summary(analysis_df, baseline)

    # STEP 4: Store results in table (canonical output)
    out_path = Path(__file__).parent / "sarr_next5_projections.csv"
    analysis_df.to_csv(out_path, index=False)
    print(f"\nProjections saved to: {out_path}")
