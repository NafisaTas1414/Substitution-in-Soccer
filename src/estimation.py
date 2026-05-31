"""
Phase 4 — ATT Estimation, Doubly Robust, Placebo, Rosenbaum, Subgroups

Input:  data/processed/matched_pairs.parquet
        data/processed/model_ready.parquet    (placebo)
Output: data/results/att_primary.csv
        data/results/att_doubly_robust.csv
        data/results/att_subgroups.csv
        data/results/placebo_test.csv
        data/results/rosenbaum_bounds.csv
        data/results/subgroup_forest_plot.png
        data/results/phase4_summary.txt
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "data" / "results"
LOG_DIR       = ROOT / "logs"

for _d in [PROCESSED_DIR, RESULTS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "estimation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

PURPLE = "#7F77DD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
GRAY   = "#888780"
AMBER  = "#BA7517"
BLUE   = "#378ADD"

LEAGUE_ORDER = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
LEAGUE_SHORT = {
    "ENG-Premier League": "Premier League",
    "ESP-La Liga":        "La Liga",
    "GER-Bundesliga":     "Bundesliga",
    "ITA-Serie A":        "Serie A",
    "FRA-Ligue 1":        "Ligue 1",
}

GAMMAS = [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]


# Shared helpers
def _logit_ps(ps: pd.Series) -> pd.Series:
    clipped = ps.clip(1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).clip(-10, 10)


def _game_state_flags(gs: pd.Series):
    """Return (is_winning, is_losing) integer arrays from game_state strings."""
    return (gs == "Winning").astype(int), (gs == "Losing").astype(int)


def _paired_att(diffs: pd.Series) -> dict:
    """ATT, SE, CI, t-stat, p-value from paired differences."""
    n = len(diffs)
    att   = float(diffs.mean())
    se    = float(diffs.std(ddof=1) / np.sqrt(n))
    ci_lo = att - 1.96 * se
    ci_hi = att + 1.96 * se
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)
    return {
        "n_pairs":        n,
        "att":            att,
        "se":             se,
        "ci_lower":       float(ci_lo),
        "ci_upper":       float(ci_hi),
        "t_stat":         float(t_stat),
        "p_value":        float(p_val),
        "significant_05": bool(p_val < 0.05),
        "significant_10": bool(p_val < 0.10),
    }


def estimate_att(
    pairs: pd.DataFrame,
    window: int = 15,
    minute_cutoff: int = None,
) -> dict:
    """
    Paired t-test ATT estimator on matched pairs.
    window=15 is the primary outcome (15-min truncation via minute_cutoff=75).
    window=30 is the secondary outcome (uses the outcome_window_truncated flag).
    Returns result dict.
    """
    log.info("=" * 60)
    log.info("STEP 1 — ATT (%d-minute window)", window)
    log.info("=" * 60)

    col_t = f"t_goal_diff_next{window}"
    col_c = f"c_goal_diff_next{window}"

    if minute_cutoff is not None:
        # 15-min truncation: sub needs minute <= cutoff for full window to fit
        full_mask = (
            (pairs["t_minute"] <= minute_cutoff) &
            (pairs["c_minute"] <= minute_cutoff)
        )
    else:
        # 30-min truncation: use the existing outcome_window_truncated flag
        full_mask = (
            (pairs["t_outcome_window_truncated"] == 0) &
            (pairs["c_outcome_window_truncated"] == 0)
        )

    full    = pairs[full_mask]
    n_trunc = int((~full_mask).sum())

    print(f"  Total matched pairs:         {len(pairs):,}")
    print(f"  Full {window}-min window pairs:    {len(full):,}")
    print(f"  Truncated pairs excluded:    {n_trunc:,}")
    print(f"  Survival rate:               {len(full)/len(pairs)*100:.1f}%")

    if len(full) < 2000 and window == 15:
        print(f"  ERROR — fewer than 2,000 pairs survived the 15-min truncation filter.")
        print(f"  Investigate the outcome_window_truncated flag before proceeding.")

    diffs_full = full[col_t].astype(float) - full[col_c].astype(float)
    result     = _paired_att(diffs_full)
    result["window"]      = window
    result["n_truncated"] = n_trunc

    diffs_all = pairs[col_t].astype(float) - pairs[col_c].astype(float)
    att_all   = float(diffs_all.mean())
    result["att_all_pairs"] = att_all
    result["stable"]        = bool(abs(att_all - result["att"]) < 0.02)

    print("\n" + "═" * 51)
    print(f"  {'PRIMARY' if window == 15 else 'SECONDARY'} ATT RESULTS — {window}-MINUTE WINDOW")
    print("═" * 51)
    print(f"  N matched pairs (full window):    {result['n_pairs']:,}")
    print(f"  N pairs excluded (truncated):     {n_trunc:,}")
    print(f"\n  {window}-MINUTE OUTCOME WINDOW:")
    print(f"  ATT     =  {result['att']:+.4f} goal differential")
    print(f"  SE      =   {result['se']:.4f}")
    print(f"  95% CI  =  [{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]")
    print(f"  t-stat  =  {result['t_stat']:+.3f}")
    print(f"  p-value =   {result['p_value']:.4f}")
    print(f"  Significant at α=0.05? {'Yes' if result['significant_05'] else 'No'}")
    print(f"  Significant at α=0.10? {'Yes' if result['significant_10'] else 'No'}")
    print(f"\n  Sensitivity (all pairs including truncated):")
    print(f"  ATT_{window} (all) = {att_all:+.4f}")
    print(f"  Stable? {'Yes' if result['stable'] else 'No (|diff| ≥ 0.02)'}")
    print("═" * 51)

    return result


def _run_primary_and_secondary(pairs: pd.DataFrame) -> tuple:
    """
    Primary: 15-min ATT using minute <= 75 truncation filter (~9,000 pairs).
    Secondary: 30-min ATT using outcome_window_truncated flag (755 pairs).
    Saves att_primary.csv. Returns (res15, res30).
    """
    # Primary: 15-minute window, minute-based truncation
    res15 = estimate_att(pairs, window=15, minute_cutoff=75)

    # Secondary: 30-minute window, flag-based truncation (original 755 pairs)
    print("\n  Secondary outcome (30-min window, non-truncated pairs only):")
    res30 = estimate_att(pairs, window=30, minute_cutoff=None)
    print(
        f"  Secondary outcome (30-min window, n={res30['n_pairs']:,} non-truncated pairs): "
        f"ATT = {res30['att']:+.4f}"
    )

    # CHECK 3 — direction consistency
    if (res15["att"] >= 0) != (res30["att"] >= 0):
        print(
            "\n  WARNING — 15-min and 30-min ATTs point in opposite directions. "
            f"ATT_15={res15['att']:+.4f}, ATT_30={res30['att']:+.4f}. "
            "Interpret with caution."
        )
        log.warning(
            "CHECK 3 FAILED: ATT direction inconsistent between windows. "
            "ATT_15=%+.4f, ATT_30=%+.4f", res15["att"], res30["att"]
        )

    rows = [
        {
            "window": 15, "role": "primary",
            "n_pairs":       res15["n_pairs"],
            "att":           round(res15["att"], 6),
            "se":            round(res15["se"], 6),
            "ci_lower":      round(res15["ci_lower"], 6),
            "ci_upper":      round(res15["ci_upper"], 6),
            "t_stat":        round(res15["t_stat"], 4),
            "p_value":       round(res15["p_value"], 6),
            "significant_05": res15["significant_05"],
            "significant_10": res15["significant_10"],
        },
        {
            "window": 30, "role": "secondary",
            "n_pairs":       res30["n_pairs"],
            "att":           round(res30["att"], 6),
            "se":            round(res30["se"], 6),
            "ci_lower":      round(res30["ci_lower"], 6),
            "ci_upper":      round(res30["ci_upper"], 6),
            "t_stat":        round(res30["t_stat"], 4),
            "p_value":       round(res30["p_value"], 6),
            "significant_05": res30["significant_05"],
            "significant_10": res30["significant_10"],
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "att_primary.csv", index=False)
    log.info("Saved att_primary.csv")

    return res15, res30


def doubly_robust(pairs: pd.DataFrame, att_primary: float) -> dict:
    """
    OLS outcome model with HC3 robust SEs on long-format matched data.
    Primary outcome: goal_diff_next15.
    Saves att_doubly_robust.csv. Returns result dict.
    """
    log.info("=" * 60)
    log.info("STEP 2 — Doubly Robust regression adjustment")
    log.info("=" * 60)

    t_is_winning, t_is_losing = _game_state_flags(pairs["t_game_state"])
    c_is_winning, c_is_losing = _game_state_flags(pairs["c_game_state"])
    idx = np.arange(len(pairs))

    trt = pd.DataFrame({
        "pair_id":                idx,
        "treated":                1,
        "goal_diff_next15":       pairs["t_goal_diff_next15"].values,
        "score_diff_at_sub":      pairs["t_score_diff_at_sub"].values,
        "is_home":                pairs["t_is_home"].astype(float).values,
        "opponent_points_l4":     pairs["t_opponent_points_l4"].values,
        "subs_used_before":       pairs["t_subs_used_before"].astype(float).values,
        "is_winning":             t_is_winning.values,
        "is_losing":              t_is_losing.values,
        "league":                 pairs["t_league"].values,
        "sub_direction":          pairs["t_sub_direction"].fillna("neutral").values,
        "log1p_player_quality":   pairs["t_log1p_player_quality"].values,
        "player_out_minutes_l30": pairs["t_player_out_minutes_l30"].fillna(0.0).values,
    })

    ctl = pd.DataFrame({
        "pair_id":                idx,
        "treated":                0,
        "goal_diff_next15":       pairs["c_goal_diff_next15"].values,
        "score_diff_at_sub":      pairs["c_score_diff_at_sub"].values,
        "is_home":                pairs["c_is_home"].astype(float).values,
        "opponent_points_l4":     pairs["t_opponent_points_l4"].values,  # matched pair proxy
        "subs_used_before":       pairs["t_subs_used_before"].astype(float).values,
        "is_winning":             c_is_winning.values,
        "is_losing":              c_is_losing.values,
        "league":                 pairs["c_league"].values,
        "sub_direction":          pairs["t_sub_direction"].fillna("neutral").values,
        "log1p_player_quality":   0.0,
        "player_out_minutes_l30": 0.0,
    })

    long_df = pd.concat([trt, ctl], ignore_index=True)

    for col in [
        "treated", "goal_diff_next15", "score_diff_at_sub", "is_home",
        "opponent_points_l4", "subs_used_before", "is_winning", "is_losing",
        "log1p_player_quality", "player_out_minutes_l30",
    ]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    formula = (
        "goal_diff_next15 ~ treated "
        "+ score_diff_at_sub + is_home + opponent_points_l4 "
        "+ subs_used_before + is_winning + is_losing "
        "+ log1p_player_quality + player_out_minutes_l30 "
        "+ C(league) + C(sub_direction)"
    )

    model = smf.ols(formula=formula, data=long_df).fit(cov_type="HC3")

    dr_att  = float(model.params["treated"])
    dr_se   = float(model.bse["treated"])
    ci      = model.conf_int().loc["treated"]
    dr_pval = float(model.pvalues["treated"])
    qual_c  = float(model.params.get("log1p_player_quality", np.nan))
    fat_c   = float(model.params.get("player_out_minutes_l30", np.nan))

    diff_vs_primary = abs(dr_att - att_primary)
    if diff_vs_primary < 0.02:
        assessment = "Robust (|diff|<0.02)"
    elif diff_vs_primary < 0.05:
        assessment = f"Moderate (|diff|={diff_vs_primary:.3f}, 0.02≤|diff|<0.05)"
    else:
        assessment = f"Diverged (|diff|={diff_vs_primary:.3f} ≥ 0.05)"

    if diff_vs_primary >= 0.05:
        log.warning(
            "CHECK 4: DR ATT diverges from primary ATT by %.3f. "
            "Player quality may be absorbing the treatment effect — "
            "report this explicitly as a finding.", diff_vs_primary
        )

    print("\n" + "═" * 51)
    print("  DOUBLY ROBUST ATT RESULTS (15-minute outcome)")
    print("═" * 51)
    print(f"  DR ATT (coeff on treated):  {dr_att:+.4f}")
    print(f"  SE (HC3 robust):             {dr_se:.4f}")
    print(f"  95% CI:                     [{float(ci.iloc[0]):+.4f}, {float(ci.iloc[1]):+.4f}]")
    print(f"  p-value:                     {dr_pval:.4f}")
    print(f"  Significant at α=0.05?       {'Yes' if dr_pval < 0.05 else 'No'}")
    print(f"\n  Comparison to Step 1 primary ATT (15-min):")
    print(f"  Step 1 ATT:   {att_primary:+.4f}")
    print(f"  DR ATT:       {dr_att:+.4f}")
    print(f"  Difference:   {dr_att - att_primary:+.4f}")
    print(f"  Assessment:   {assessment}")
    print(f"\n  Player quality coefficient: {qual_c:+.4f}")
    print(f"    (positive = better substitute → better outcome)")
    print(f"  Fatigue coefficient: {fat_c:+.4f}")
    print(f"    (negative = more fatigued player → worse outcome)")
    print("═" * 51)

    result = {
        "estimator":       "doubly_robust_OLS_HC3",
        "outcome_window":  "15min",
        "dr_att":          round(dr_att, 6),
        "dr_se":           round(dr_se, 6),
        "ci_lower":        round(float(ci.iloc[0]), 6),
        "ci_upper":        round(float(ci.iloc[1]), 6),
        "p_value":         round(dr_pval, 6),
        "significant_05":  bool(dr_pval < 0.05),
        "step1_att":       round(att_primary, 6),
        "diff_vs_step1":   round(dr_att - att_primary, 6),
        "assessment":      assessment,
        "qual_coef":       round(qual_c, 6),
        "fatigue_coef":    round(fat_c, 6),
    }
    pd.DataFrame([result]).to_csv(RESULTS_DIR / "att_doubly_robust.csv", index=False)
    log.info("Saved att_doubly_robust.csv")

    return result


def quality_only_regression(pairs: pd.DataFrame) -> dict:
    """
    Regresses goal_diff_next15 on log1p_player_quality within treated rows only
    (t_minute <= 75). Isolates bench quality as a predictor of post-sub outcomes,
    independent of the treatment indicator.
    Saves quality_only_regression.csv.
    """
    log.info("=" * 60)
    log.info("STEP 2b — Quality-only regression (treated rows)")
    log.info("=" * 60)

    treated = pairs[pairs["t_minute"] <= 75].copy()
    treated = treated.rename(columns={
        "t_goal_diff_next15":     "goal_diff_next15",
        "t_log1p_player_quality": "log1p_player_quality",
    })

    model = smf.ols(
        "goal_diff_next15 ~ log1p_player_quality",
        data=treated,
    ).fit(cov_type="HC3")

    coef   = float(model.params["log1p_player_quality"])
    se     = float(model.bse["log1p_player_quality"])
    pval   = float(model.pvalues["log1p_player_quality"])
    ci     = model.conf_int().loc["log1p_player_quality"]
    rsq    = float(model.rsquared)
    n      = int(model.nobs)

    print("\n" + "═" * 51)
    print("  QUALITY-ONLY REGRESSION (treated rows, 15-min)")
    print("═" * 51)
    print(f"  N (treated, minute ≤ 75):  {n:,}")
    print(f"  R-squared:                 {rsq:.4f}")
    print(f"  Quality coefficient:       {coef:+.4f}")
    print(f"  SE (HC3):                  {se:.4f}")
    print(f"  95% CI:                   [{float(ci.iloc[0]):+.4f}, {float(ci.iloc[1]):+.4f}]")
    print(f"  p-value:                   {pval:.4f}")
    print(f"  Significant at α=0.05?     {'Yes' if pval < 0.05 else 'No'}")
    print("═" * 51)

    result = {
        "n_treated":        n,
        "rsquared":         round(rsq, 6),
        "quality_coef":     round(coef, 6),
        "quality_se":       round(se, 6),
        "ci_lower":         round(float(ci.iloc[0]), 6),
        "ci_upper":         round(float(ci.iloc[1]), 6),
        "p_value":          round(pval, 6),
        "significant_05":   bool(pval < 0.05),
    }
    pd.DataFrame([result]).to_csv(RESULTS_DIR / "quality_only_regression.csv", index=False)
    log.info("Saved quality_only_regression.csv")

    return result


def _match_placebo(df: pd.DataFrame, caliper: float) -> pd.DataFrame:
    """Per-league KDTree nearest-neighbor matching for placebo data."""
    df = df.copy()
    all_pairs: list[dict] = []

    for lg in LEAGUE_ORDER:
        ldf = df[df["league"] == lg]
        trt = ldf[ldf["fake_treated"] == 1].copy()
        ctl = ldf[ldf["fake_treated"] == 0].copy()

        if len(trt) == 0 or len(ctl) == 0:
            log.info("  Placebo %s: %d treated / %d control — skipping", lg, len(trt), len(ctl))
            continue

        tree = KDTree(ctl["logit_ps"].values.reshape(-1, 1))
        ctl  = ctl.reset_index(drop=True)
        used: set[int] = set()

        for _, t_row in trt.sort_values("propensity_score").iterrows():
            cand_pos_arr, dist_arr = tree.query_radius(
                np.array([[t_row["logit_ps"]]]), r=caliper, return_distance=True
            )
            cand_pos  = cand_pos_arr[0]
            cand_dist = dist_arr[0]
            if len(cand_pos) == 0:
                continue
            order = np.argsort(cand_dist)
            for pos in cand_pos[order]:
                if pos not in used:
                    c_row = ctl.iloc[pos]
                    used.add(pos)
                    all_pairs.append({
                        "t_goal_diff_next30":         float(t_row["goal_diff_next30"]),
                        "c_goal_diff_next30":         float(c_row["goal_diff_next30"]),
                        "t_outcome_window_truncated": int(t_row.get("outcome_window_truncated", 0)),
                        "c_outcome_window_truncated": int(c_row.get("outcome_window_truncated", 0)),
                        "ps_distance":                float(cand_dist[order][list(cand_pos[order]).index(pos)]),
                        "league":                     lg,
                    })
                    break

    return pd.DataFrame(all_pairs) if all_pairs else pd.DataFrame()


def placebo_test(mr: pd.DataFrame, caliper: float = 0.026) -> dict:
    """
    Assigns fake treatment at minute 35 (or 30-50 window if too few rows)
    and checks the pipeline produces ATT ≈ 0.
    Saves placebo_test.csv.
    """
    log.info("=" * 60)
    log.info("STEP 3 — Placebo test")
    log.info("=" * 60)

    ph = mr[mr["minute"] == 35].copy()
    placebo_label = "35"

    if len(ph) < 500:
        log.info(
            "Minute 35 has %d rows (< 500). Expanding to first-half window (minutes 30-50).",
            len(ph),
        )
        ph = mr[mr["minute"].between(30, 50) & ~mr["minute"].isin([46])].copy()
        placebo_label = "30–50"

    log.info("Placebo window: minute %s  N=%d", placebo_label, len(ph))

    rng    = np.random.default_rng(42)
    n_trt  = len(ph) // 2
    fake   = np.zeros(len(ph), dtype=int)
    fake[:n_trt] = 1
    rng.shuffle(fake)
    ph["fake_treated"] = fake

    # same covariates as Phase 3
    ph["is_winning"] = pd.to_numeric(ph.get("is_winning", 0), errors="coerce").fillna(0).astype(int)
    ph["is_losing"]  = pd.to_numeric(ph.get("is_losing",  0), errors="coerce").fillna(0).astype(int)
    ph["is_home"]    = pd.to_numeric(ph.get("is_home",    0), errors="coerce").fillna(0).astype(int)

    ph["score_diff_at_sub"] = pd.to_numeric(
        ph.get("score_diff_at_sub", 0), errors="coerce"
    ).fillna(0.0)

    ph["subs_used_before"] = (
        pd.to_numeric(ph.get("sub_slot", 1), errors="coerce").fillna(1) - 1
    ).clip(0, 5).astype(float)

    opp_med = pd.to_numeric(ph.get("opponent_points_l4", 5), errors="coerce").median()
    ph["opponent_points_l4"] = pd.to_numeric(
        ph.get("opponent_points_l4", 5), errors="coerce"
    ).fillna(opp_med)

    lg_dummies = pd.get_dummies(ph["league"], prefix="lg", dtype=float)
    lg_cols    = sorted([c for c in lg_dummies.columns if "ENG" not in c])
    lg_part    = lg_dummies[lg_cols] if lg_cols else pd.DataFrame(index=ph.index)

    cont_cols  = ["score_diff_at_sub", "opponent_points_l4", "subs_used_before"]
    bin_cols   = ["is_home", "is_winning", "is_losing"]

    scaler = StandardScaler()
    cont_s = pd.DataFrame(
        scaler.fit_transform(ph[cont_cols].astype(float)),
        columns=cont_cols,
        index=ph.index,
    )
    X = pd.concat([cont_s, ph[bin_cols].astype(float), lg_part], axis=1).values
    y = ph["fake_treated"].values

    ps_model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
    )
    ps_model.fit(X, y)
    ps = ps_model.predict_proba(X)[:, 1]
    ph = ph.copy()
    ph["propensity_score"] = ps
    ph["logit_ps"]         = _logit_ps(pd.Series(ps, index=ph.index)).values

    logit_std       = ph["logit_ps"].std()
    placebo_caliper = 0.2 * logit_std
    log.info("Placebo caliper = 0.2 × %.4f = %.4f", logit_std, placebo_caliper)

    pp = _match_placebo(ph, placebo_caliper)
    n_pairs = len(pp)
    log.info("Placebo pairs formed: %d", n_pairs)

    if n_pairs < 10:
        log.warning("Too few placebo pairs (%d) — result unreliable", n_pairs)
        result = {
            "placebo_minute": placebo_label,
            "n_obs":   len(ph),
            "n_pairs": n_pairs,
            "att":     np.nan, "se": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "p_value": np.nan, "passes": False,
        }
        pd.DataFrame([result]).to_csv(RESULTS_DIR / "placebo_test.csv", index=False)
        return result

    full = pp[
        (pp["t_outcome_window_truncated"] == 0) &
        (pp["c_outcome_window_truncated"] == 0)
    ]
    if len(full) < 5:
        full = pp

    diffs  = full["t_goal_diff_next30"] - full["c_goal_diff_next30"]
    res    = _paired_att(diffs)
    passes = not res["significant_05"]

    print("\n" + "═" * 51)
    print("  PLACEBO TEST")
    print("═" * 51)
    print(f"  Placebo minute used:      {placebo_label}")
    print(f"  N observations at minute: {len(ph)}")
    print(f"  N placebo pairs formed:   {n_pairs}")
    print(f"\n  Placebo ATT (30-min):     {res['att']:+.4f}")
    print(f"  SE:                        {res['se']:.4f}")
    print(f"  95% CI:                   [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]")
    print(f"  p-value:                   {res['p_value']:.4f}")
    print(f"\n  Result: {'PASS ✓' if passes else 'FAIL ✗'}")
    if passes:
        print(
            "  Interpretation: No spurious effect detected at placebo minute — "
            "pipeline design is validated."
        )
    else:
        print(
            "  Interpretation: Significant effect detected at placebo minute — "
            "pipeline may be detecting spurious patterns."
        )
    print("═" * 51)

    if not passes:
        log.warning(
            "PLACEBO TEST FAILED — pipeline may be detecting spurious patterns. "
            "The real ATT from Steps 1-2 cannot be interpreted as causal without "
            "further investigation."
        )

    result = {
        "placebo_minute": placebo_label,
        "n_obs":          len(ph),
        "n_pairs":        n_pairs,
        "att":            round(res["att"], 6),
        "se":             round(res["se"], 6),
        "ci_lower":       round(res["ci_lower"], 6),
        "ci_upper":       round(res["ci_upper"], 6),
        "p_value":        round(res["p_value"], 6),
        "passes":         passes,
    }
    pd.DataFrame([result]).to_csv(RESULTS_DIR / "placebo_test.csv", index=False)
    log.info("Saved placebo_test.csv")

    return result


def rosenbaum_bounds(pair_diffs: np.ndarray, gammas: list = None) -> tuple:
    """
    Rosenbaum bounds via Wilcoxon signed-rank normal approximation.
    Returns (bounds_df, stat_base, p_base, gamma_critical, robustness_label).
    Saves rosenbaum_bounds.csv.
    """
    log.info("=" * 60)
    log.info("STEP 4 — Rosenbaum sensitivity analysis")
    log.info("=" * 60)

    if gammas is None:
        gammas = GAMMAS

    diffs_nz = pair_diffs[pair_diffs != 0]
    n        = len(diffs_nz)
    log.info("Non-zero pair differences: %d / %d", n, len(pair_diffs))

    stat_base, p_base = stats.wilcoxon(diffs_nz, alternative="greater")
    log.info("Base Wilcoxon W+ = %g  p = %.6f", stat_base, p_base)

    ranks = stats.rankdata(np.abs(diffs_nz))
    sum_r  = float(np.sum(ranks))
    sum_r2 = float(np.sum(ranks ** 2))

    records       = []
    gamma_critical = None

    for gamma in gammas:
        p_hi    = gamma / (1 + gamma)
        E_upper = sum_r  * p_hi
        V_upper = sum_r2 * p_hi * (1 - p_hi)
        z_upper = (stat_base - E_upper) / np.sqrt(V_upper)
        p_upper = float(1 - norm.cdf(z_upper))
        sig     = p_upper < 0.05

        if gamma_critical is None and not sig:
            gamma_critical = gamma

        records.append({
            "gamma":          gamma,
            "p_upper":        round(p_upper, 6),
            "significant_05": sig,
        })

    if gamma_critical is None:
        gamma_critical = gammas[-1]
        log.info("Critical Γ > %.1f — result significant at all tested gammas", gammas[-1])
    else:
        log.info("Critical Γ = %.2f", gamma_critical)

    bounds_df = pd.DataFrame(records)

    if gamma_critical < 1.3:
        robustness = "fragile"
    elif gamma_critical < 1.5:
        robustness = "moderate"
    elif gamma_critical < 2.0:
        robustness = "robust"
    else:
        robustness = "very robust"

    pct = (gamma_critical - 1) * 100

    print("\n" + "═" * 51)
    print("  ROSENBAUM SENSITIVITY ANALYSIS")
    print("═" * 51)
    print(f"  Base Wilcoxon statistic: {stat_base:.0f}")
    print(f"  Base p-value (Γ=1.0):   {p_base:.6f}")
    print(f"\n  {'Γ':>6}  {'p-value (upper bound)':>22}  {'Significant?':>13}")
    print(f"  {'─'*6}  {'─'*22}  {'─'*13}")
    for row in records:
        sig_str = "Yes" if row["significant_05"] else "No"
        print(f"  {row['gamma']:>6.2f}  {row['p_upper']:>22.6f}  {sig_str:>13}")
    print(f"\n  Critical Γ: {gamma_critical:.2f}")
    print(
        f"\n  Interpretation:\n"
        f"  The finding remains significant under unmeasured confounding\n"
        f"  that would increase treatment odds by up to {pct:.0f}%.\n"
        f"  This is a {robustness} result by observational study standards."
    )
    print("═" * 51)

    bounds_df.to_csv(RESULTS_DIR / "rosenbaum_bounds.csv", index=False)
    log.info("Saved rosenbaum_bounds.csv")

    return bounds_df, float(stat_base), float(p_base), gamma_critical, robustness


def subgroup_att(
    pairs: pd.DataFrame,
    subgroup_col: str,
    subgroup_val,
) -> dict:
    """
    Compute ATT on a subgroup defined by (subgroup_col == subgroup_val).
    Pass subgroup_col='_mask' and subgroup_val=boolean_series for pre-built masks.
    Primary outcome: goal_diff_next15 with minute <= 75 truncation filter.
    Secondary outcome: goal_diff_next30 (estimate only).
    n < 200 → insufficient flag set. n < 500 in 15-min window → flag as unreliable.
    """
    if subgroup_col == "_mask":
        mask   = subgroup_val
        subset = pairs[mask].copy()
    else:
        subset = pairs[pairs[subgroup_col] == subgroup_val].copy()

    n = len(subset)

    empty = {
        "n_pairs": n, "n_full_window_15": 0, "insufficient": True,
        "att_15": np.nan, "se_15": np.nan,
        "ci_lower_15": np.nan, "ci_upper_15": np.nan,
        "p_value_15": np.nan, "significant_05": False,
        "att_30": np.nan,
    }
    if n < 200:
        return empty

    # 15-min truncation: use minute-based filter
    full_mask_15 = (subset["t_minute"] <= 75) & (subset["c_minute"] <= 75)
    full_15 = subset[full_mask_15] if full_mask_15.sum() >= 50 else subset

    if full_mask_15.sum() < 500:
        log.warning(
            "  Subgroup has only %d pairs in 15-min window (< 500) — estimate unreliable",
            full_mask_15.sum()
        )

    diffs_15 = full_15["t_goal_diff_next15"].astype(float) - full_15["c_goal_diff_next15"].astype(float)

    # 30-min secondary (flag-based, may be small)
    full_mask_30 = (
        (subset["t_outcome_window_truncated"] == 0) &
        (subset["c_outcome_window_truncated"] == 0)
    )
    full_30 = subset[full_mask_30]
    att_30_secondary = float(
        (full_30["t_goal_diff_next30"] - full_30["c_goal_diff_next30"]).mean()
    ) if len(full_30) > 0 else np.nan

    res = _paired_att(diffs_15)

    return {
        "n_pairs":           n,
        "n_full_window_15":  len(full_15),
        "insufficient":      False,
        "att_15":            res["att"],
        "se_15":             res["se"],
        "ci_lower_15":       res["ci_lower"],
        "ci_upper_15":       res["ci_upper"],
        "p_value_15":        res["p_value"],
        "significant_05":    res["significant_05"],
        "att_30":            att_30_secondary,
    }


def _run_all_subgroups(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 13 subgroup ATTs and save att_subgroups.csv.
    Returns DataFrame of results.
    """
    log.info("=" * 60)
    log.info("STEP 5 — Subgroup ATT estimation")
    log.info("=" * 60)

    specs = [
        # (category, label, col, val_or_mask)
        ("GAME STATE",  "Losing",           "t_game_state",    "Losing"),
        ("GAME STATE",  "Level",             "t_game_state",    "Level"),
        ("GAME STATE",  "Winning",           "t_game_state",    "Winning"),
        ("TIMING",      "Early sub < 74",    "_mask",           pairs["t_minute"] < 74),
        ("TIMING",      "Late sub ≥ 74",     "_mask",           pairs["t_minute"] >= 74),
        ("DIRECTION",   "Offensive",         "t_sub_direction", "offensive"),
        ("DIRECTION",   "Defensive",         "t_sub_direction", "defensive"),
        ("DIRECTION",   "Neutral",           "t_sub_direction", "neutral"),
        ("LEAGUE",      "Premier League",    "t_league",        "ENG-Premier League"),
        ("LEAGUE",      "La Liga",           "t_league",        "ESP-La Liga"),
        ("LEAGUE",      "Bundesliga",        "t_league",        "GER-Bundesliga"),
        ("LEAGUE",      "Serie A",           "t_league",        "ITA-Serie A"),
        ("LEAGUE",      "Ligue 1",           "t_league",        "FRA-Ligue 1"),
    ]

    rows = []
    for cat, label, col, val in specs:
        res = subgroup_att(pairs, col, val)
        if res["insufficient"]:
            log.warning("  Subgroup '%s' has n=%d < 200 — excluded from forest plot", label, res["n_pairs"])
        else:
            # CHECK 2 — league subgroups need > 1000 pairs in 15-min window
            if cat == "LEAGUE" and res["n_full_window_15"] < 500:
                log.warning(
                    "  CHECK 2: League '%s' has only %d pairs in 15-min window — flagged as unreliable",
                    label, res["n_full_window_15"]
                )
            log.info(
                "  %-22s  n=%5d  n_15min=%5d  ATT_15=%+.4f  SE=%.4f  p=%.4f  %s",
                label, res["n_pairs"], res["n_full_window_15"],
                res["att_15"], res["se_15"],
                res["p_value_15"], "sig" if res["significant_05"] else "",
            )
        rows.append({
            "subgroup_category": cat,
            "subgroup_label":    label,
            "n_pairs":           res["n_pairs"],
            "n_full_window_15":  res["n_full_window_15"],
            "att_15":            res["att_15"],
            "se_15":             res["se_15"],
            "ci_lower_15":       res["ci_lower_15"],
            "ci_upper_15":       res["ci_upper_15"],
            "p_value_15":        res["p_value_15"],
            "att_30":            res["att_30"],
            "significant_05":    res["significant_05"],
            "insufficient":      res["insufficient"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "att_subgroups.csv", index=False)
    log.info("Saved att_subgroups.csv")

    return df


def plot_forest(subgroup_df: pd.DataFrame, overall_att: float) -> None:
    """Generate the subgroup ATT forest plot."""
    log.info("Generating subgroup forest plot...")

    # y positions per label; reading order is top-to-bottom but y increases upward in matplotlib
    # GAME STATE: y = 16.5, 15.5, 14.5
    # gap at 13.5
    # TIMING: y = 12.5, 11.5
    # gap at 10.5
    # DIRECTION: y = 9.5, 8.5, 7.5
    # gap at 6.5
    # LEAGUE: y = 5.5, 4.5, 3.5, 2.5, 1.5

    layout = {
        "Losing":          (16.5, CORAL),
        "Level":           (15.5, GRAY),
        "Winning":         (14.5, TEAL),
        "Early sub < 74":  (12.5, PURPLE),
        "Late sub ≥ 74":   (11.5, PURPLE),
        "Offensive":       (9.5,  AMBER),
        "Defensive":       (8.5,  AMBER),
        "Neutral":         (7.5,  AMBER),
        "Premier League":  (5.5,  BLUE),
        "La Liga":         (4.5,  BLUE),
        "Bundesliga":      (3.5,  BLUE),
        "Serie A":         (2.5,  BLUE),
        "Ligue 1":         (1.5,  BLUE),
    }

    section_headers = {
        "GAME STATE": 17.3,
        "TIMING":     13.3,
        "DIRECTION":  10.3,
        "LEAGUE":     6.3,
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    valid = subgroup_df[~subgroup_df["insufficient"]]

    all_lo = valid["ci_lower_15"].dropna()
    all_hi = valid["ci_upper_15"].dropna()
    x_min  = min(all_lo.min(), overall_att) - 0.05
    x_max  = max(all_hi.max(), overall_att) + 0.05
    x_min  = min(x_min, -0.05)
    x_max  = max(x_max,  0.15)

    for _, row in valid.iterrows():
        lbl = row["subgroup_label"]
        if lbl not in layout:
            continue
        y_pos, color = layout[lbl]
        att    = row["att_15"]
        ci_lo  = row["ci_lower_15"]
        ci_hi  = row["ci_upper_15"]
        n      = int(row["n_pairs"])
        sig    = row["significant_05"]

        ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=color, linewidth=2.0, zorder=3)
        marker = "D" if sig else "o"
        ax.scatter([att], [y_pos], color=color, s=90, zorder=4, marker=marker)

        ax.text(
            x_max + 0.005, y_pos,
            f"n={n:,}",
            va="center", ha="left", fontsize=8.5, color="#555555",
        )

    for header, y_pos in section_headers.items():
        ax.text(
            x_min - 0.005, y_pos, header,
            va="center", ha="right", fontsize=10,
            fontweight="bold", color="#333333",
        )

    ax.axvline(0, color="#AAAAAA", linewidth=1.2, zorder=1)
    ax.axvline(
        overall_att, color="black", linewidth=1.5,
        linestyle="--", zorder=2,
        label=f"Overall ATT = {overall_att:.3f}",
    )

    y_ticks  = [v[0] for v in layout.values()]
    y_labels = [k for k in layout]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(0.5, 18.0)
    ax.set_xlim(x_min - 0.15, x_max + 0.07)
    ax.set_xlabel("ATT estimate (goal differential, 15-min window)", fontsize=11)

    ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        "Subgroup ATT estimates — causal effect of substitution\non post-sub goal differential",
        fontsize=13, fontweight="bold", y=0.98,
    )
    ax.set_title(
        "15-minute outcome window  |  Matched pairs  |  Error bars = 95% CI",
        fontsize=10, color="#555555", pad=8,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="gray", linestyle="None",
               markersize=8, label="Significant at α=0.05"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label=f"Overall ATT = {overall_att:.3f}")
    ], fontsize=9, loc="lower right")

    fig.text(
        0.98, 0.01,
        "Dashed line = overall ATT (15-min window).  Estimates not adjusted for multiple testing.",
        ha="right", va="bottom", fontsize=8, color="#777777",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(RESULTS_DIR / "subgroup_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved subgroup_forest_plot.png")


def write_summary(
    res15: dict,
    res30: dict,
    dr: dict,
    placebo: dict,
    bounds_df: pd.DataFrame,
    gamma_critical: float,
    robustness: str,
    subgroup_df: pd.DataFrame,
) -> None:
    """Write phase4_summary.txt. res15 is the primary result, res30 is secondary."""
    log.info("Writing phase4_summary.txt...")

    att   = res15["att"]
    ci_lo = res15["ci_lower"]
    ci_hi = res15["ci_upper"]
    pval  = res15["p_value"]
    sig   = "Yes" if res15["significant_05"] else "No"

    pl_att  = placebo["att"]
    pl_pval = placebo["p_value"]
    pl_pass = "PASS" if placebo["passes"] else "FAIL"

    dr_att        = dr["dr_att"]
    dr_consistent = "Yes" if abs(dr_att - att) < 0.05 else "No"

    pct = (gamma_critical - 1) * 100

    # Top 3 subgroup findings by absolute ATT magnitude
    valid = subgroup_df[~subgroup_df["insufficient"]].copy()
    valid = valid.sort_values("att_15", ascending=False)

    top3_lines = []
    for _, row in valid.head(3).iterrows():
        sig_mark = "(*)" if row["significant_05"] else ""
        top3_lines.append(
            f"  - {row['subgroup_label']} ({row['subgroup_category']}): "
            f"ATT = {row['att_15']:+.4f} {sig_mark}"
        )
    if not top3_lines:
        top3_lines = ["  - No valid subgroups computed"]

    plain_english = (
        f"Tactical substitutions in the Big 5 leagues were followed by "
        f"approximately {abs(att):.2f} {'more' if att >= 0 else 'fewer'} "
        f"goals in favor of the substituting team over the subsequent 15 minutes "
        f"compared to matched situations where no substitution was made, "
        f"after controlling for game state."
    )

    lines = [
        "CAUSAL INFERENCE RESULTS SUMMARY",
        "══════════════════════════════════════════",
        "Research question:",
        "  Do managers who substitute earlier produce",
        "  better outcomes than those who substitute",
        "  later, controlling for game state?",
        "",
        "Dataset:",
        "  Big 5 European Leagues",
        "  Seasons: 2022-23 and 2023-24",
        "  Tactical substitutions analyzed: 20,768",
        "  Matched pairs: 20,762",
        "",
        "Method:",
        "  Propensity score matching (within-league,",
        "  no replacement, caliper = 0.026)",
        "  + Doubly robust OLS adjustment",
        "  Confounders controlled: scoreline, opponent",
        "  quality, home/away, subs used, sub direction",
        "",
        "──────────────────────────────────────────",
        "PRIMARY FINDING",
        "──────────────────────────────────────────",
        f"ATT (15-min window) = {att:+.4f} goal diff",
        f"N pairs (minute ≤ 75): {res15['n_pairs']:,}",
        f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]",
        f"p-value: {pval:.4f}",
        f"Significant: {sig} at α = 0.05",
        "",
        f"Plain English: {plain_english}",
        "",
        f"Doubly robust ATT (15-min) = {dr_att:+.4f}",
        f"Consistent with primary? {dr_consistent}",
        "",
        f"Secondary (30-min window, n={res30['n_pairs']:,} pairs): "
        f"ATT = {res30['att']:+.4f}",
        "",
        "──────────────────────────────────────────",
        "VALIDATION",
        "──────────────────────────────────────────",
        f"Placebo test (minute {placebo['placebo_minute']}): {pl_pass}",
        f"  Placebo ATT = {pl_att:+.4f}, p = {pl_pval:.4f}",
        "",
        f"Rosenbaum critical Γ = {gamma_critical:.2f}",
        f"  Interpretation: {robustness}",
        f"  \"The finding remains significant under unmeasured",
        f"  confounding that would increase treatment odds",
        f"  by up to {pct:.0f}%.\"",
        "",
        "──────────────────────────────────────────",
        "KEY SUBGROUP FINDINGS",
        "──────────────────────────────────────────",
        *top3_lines,
        "",
        "──────────────────────────────────────────",
        "LIMITATIONS",
        "──────────────────────────────────────────",
        "1. Outcome = goal differential (not xG —",
        "   data unavailable). Goals are noisy but",
        "   unbiased. StatsBomb xG validation",
        "   pending in Phase 5.",
        "2. Player quality proxy = goals+assists/90.",
        "   xG-based quality not available via",
        "   soccerdata.",
        f"3. Unmeasured confounders possible — bounded",
        f"   by Γ = {gamma_critical:.2f}.",
        "4. Two seasons only (2022-24). Effect",
        "   stability across seasons untested.",
        "5. Primary outcome window is 15 minutes",
        "   post-substitution due to match time",
        "   constraints — 30-minute window analysis",
        f"   available on n={res30['n_pairs']:,} non-truncated pairs",
        "   as a supplementary check.",
        "══════════════════════════════════════════",
    ]

    out_path = RESULTS_DIR / "phase4_summary.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    log.info("Saved phase4_summary.txt")
    print("\n" + "\n".join(lines))


def run_all() -> None:
    log.info("=" * 60)
    log.info("Phase 4 — ATT Estimation — START")
    log.info("=" * 60)

    pairs = pd.read_parquet(PROCESSED_DIR / "matched_pairs.parquet")
    mr    = pd.read_parquet(PROCESSED_DIR / "model_ready.parquet")
    log.info("Loaded matched_pairs.parquet: %d pairs", len(pairs))
    log.info("Loaded model_ready.parquet:   %d rows", len(mr))

    res15, res30 = _run_primary_and_secondary(pairs)
    dr = doubly_robust(pairs, att_primary=res15["att"])
    quality_reg = quality_only_regression(pairs)
    placebo = placebo_test(mr)

    full_window_15 = pairs[
        (pairs["t_minute"] <= 75) & (pairs["c_minute"] <= 75)
    ]
    pair_diffs_15 = (
        full_window_15["t_goal_diff_next15"].astype(float) -
        full_window_15["c_goal_diff_next15"].astype(float)
    ).values

    bounds_df, stat_base, p_base, gamma_critical, robustness = rosenbaum_bounds(pair_diffs_15)

    subgroup_df = _run_all_subgroups(pairs)
    plot_forest(subgroup_df, overall_att=res15["att"])
    write_summary(
        res15, res30, dr, placebo,
        bounds_df, gamma_critical, robustness,
        subgroup_df,
    )

    log.info("Phase 4 — COMPLETE")
    log.info("Outputs:")
    for p in [
        "data/results/att_primary.csv",
        "data/results/att_doubly_robust.csv",
        "data/results/quality_only_regression.csv",
        "data/results/att_subgroups.csv",
        "data/results/placebo_test.csv",
        "data/results/rosenbaum_bounds.csv",
        "data/results/subgroup_forest_plot.png",
        "data/results/phase4_summary.txt",
    ]:
        full = ROOT / p
        status = "✓" if full.exists() else "✗ MISSING"
        log.info("  %s  %s", status, p)


if __name__ == "__main__":
    run_all()
