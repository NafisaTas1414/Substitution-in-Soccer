"""
Phase 5 — Causal Forest (Heterogeneous Treatment Effect Estimation)

Fits a Regression Forest on matched pair differences to estimate
Conditional Average Treatment Effects (CATEs). Tests whether the
null average ATT from Phase 4 masks heterogeneous subgroup effects.
Uses SHAP to explain which covariates drive CATE variation.

Inputs:
  data/processed/matched_pairs.parquet

Outputs:
  data/results/cate_estimates.parquet
  data/results/cate_distribution.png
  data/results/shap_summary.png
  data/results/shap_dependence_plots.png
  data/results/heterogeneity_test.csv
  data/results/phase5_summary.txt
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from econml.grf import RegressionForest
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
PHASE4_ATT = -0.0143

LEAGUE_ENCODE = {
    "ENG-Premier League": 0,
    "ESP-La Liga":        1,
    "GER-Bundesliga":     2,
    "ITA-Serie A":        3,
    "FRA-Ligue 1":        4,
}
LEAGUE_LABELS = {v: k for k, v in LEAGUE_ENCODE.items()}

DIRECTION_ENCODE = {
    "offensive": 0,
    "defensive": 1,
    "neutral":   2,
    "unknown":   3,
}

FEATURE_NAMES = [
    "score_diff_at_sub",
    "is_home",
    "opponent_points_l4",
    "subs_used_before",
    "is_winning",
    "is_losing",
    "t_minute",
    "log1p_player_quality",
    "player_out_minutes_l30",
    "league_encoded",
    "sub_direction_encoded",
]

# subgroup Phase 4 ATT reference values (from phase4_summary.txt)
PHASE4_SUBGROUP_ATT = {
    "Losing":             -0.0529,
    "Level":               0.0212,
    "Winning":             0.0095,
    "Early sub < 74":     -0.0120,
    "Late sub >= 74":     -0.0367,
    "Offensive":          -0.0347,
    "Defensive":          -0.0330,
    "Neutral":             0.0000,
    "ENG-Premier League": -0.0248,
    "ESP-La Liga":        -0.0380,
    "GER-Bundesliga":     -0.0140,
    "ITA-Serie A":         0.0264,
    "FRA-Ligue 1":        -0.0309,
}


# STEP 1 — Prepare data
def prepare_data(pairs: pd.DataFrame) -> tuple:
    """
    Apply 15-min truncation (same as Phase 4 primary sample),
    build feature matrix X and outcome vector Y_diff.
    Returns (X, Y_diff, meta_df).
    """
    log.info("=" * 60)
    log.info("STEP 1 — Prepare causal forest dataset")
    log.info("=" * 60)

    # same filter as Phase 4 primary — keeps ~9,070 pairs
    cf = pairs[
        (pairs["t_minute"] <= 75) & (pairs["c_minute"] <= 75)
    ].copy()

    # derive binary game-state indicators from t_game_state
    cf["is_winning"] = (cf["t_game_state"] == "Winning").astype(float)
    cf["is_losing"]  = (cf["t_game_state"] == "Losing").astype(float)

    # encode categoricals
    cf["league_encoded"] = (
        cf["t_league"].map(LEAGUE_ENCODE).fillna(0).astype(int)
    )
    cf["sub_direction_encoded"] = (
        cf["t_sub_direction"]
        .str.lower()
        .map(DIRECTION_ENCODE)
        .fillna(3)
        .astype(int)
    )

    # outcome: matched pair difference
    cf["Y_diff"] = (
        cf["t_goal_diff_next15"].astype(float) -
        cf["c_goal_diff_next15"].astype(float)
    )

    feature_cols = {
        "score_diff_at_sub":    "t_score_diff_at_sub",
        "is_home":              "t_is_home",
        "opponent_points_l4":   "t_opponent_points_l4",
        "subs_used_before":     "t_subs_used_before",
        "is_winning":           "is_winning",
        "is_losing":            "is_losing",
        "t_minute":             "t_minute",
        "log1p_player_quality": "t_log1p_player_quality",
        "player_out_minutes_l30": "t_player_out_minutes_l30",
        "league_encoded":       "league_encoded",
        "sub_direction_encoded":"sub_direction_encoded",
    }

    X_df = pd.DataFrame({
        feat: cf[col].astype(float)
        for feat, col in feature_cols.items()
    })

    # fill any residual NaNs with column medians
    missing_total = X_df.isna().sum().sum()
    if missing_total > 0:
        X_df = X_df.fillna(X_df.median())
        log.warning("Filled %d missing values with column medians", missing_total)

    X       = X_df.values
    Y_diff  = cf["Y_diff"].values

    # metadata kept alongside for later steps
    meta = cf[["t_league", "t_minute", "t_game_state", "t_sub_direction"]].copy()
    meta = meta.reset_index(drop=True)

    n = len(X)
    print(f"\n  N observations for causal forest: {n:,}")
    print(f"  Mean Y (should ≈ Phase 4 ATT {PHASE4_ATT}): {Y_diff.mean():.4f}")
    print(f"  Std Y:    {Y_diff.std():.4f}")
    print(f"  X shape:  {X.shape[0]} × {X.shape[1]}")
    print(f"  Missing values in X: {missing_total}")

    return X_df, Y_diff, meta


# STEP 2 — Fit the causal forest
def fit_forest(X_df: pd.DataFrame, Y_diff: np.ndarray) -> tuple:
    """
    Fit a RegressionForest on (X, Y_diff) to estimate CATEs.
    Returns (forest, cate_estimates, cate_lower, cate_upper).
    """
    log.info("=" * 60)
    log.info("STEP 2 — Fit causal forest (n_estimators=2000)")
    log.info("=" * 60)

    X = X_df.values

    forest = RegressionForest(
        n_estimators=2000,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    log.info("Fitting forest on %d observations × %d features ...", *X.shape)
    forest.fit(X, Y_diff)
    log.info("Forest fit complete")

    cate_estimates = forest.predict(X).ravel()

    # 95% confidence intervals
    pred_interval = forest.predict_interval(X, alpha=0.05)
    cate_lower = pred_interval[0].ravel()
    cate_upper = pred_interval[1].ravel()

    mean_cate = cate_estimates.mean()
    diff_vs_p4 = abs(mean_cate - PHASE4_ATT)

    print("\n" + "═" * 51)
    print("  CAUSAL FOREST — CATE SUMMARY")
    print("═" * 51)
    print(f"  Mean CATE:   {mean_cate:+.4f}  (should ≈ {PHASE4_ATT})")
    print(f"  Std CATE:    {cate_estimates.std():.4f}")
    print(f"  Min CATE:    {cate_estimates.min():+.4f}")
    print(f"  Max CATE:    {cate_estimates.max():+.4f}")
    print(f"  % CATE > 0:  {(cate_estimates > 0).mean()*100:.1f}%")
    print(f"  % CATE < 0:  {(cate_estimates < 0).mean()*100:.1f}%")
    print()
    print(f"  Consistency check with Phase 4:")
    print(f"  Phase 4 ATT:             {PHASE4_ATT}")
    print(f"  Causal forest mean CATE: {mean_cate:.4f}")
    print(f"  Difference:              {diff_vs_p4:.4f}")
    print(f"  Consistent?              {'Yes' if diff_vs_p4 < 0.05 else 'No'}")
    print("═" * 51)

    return forest, cate_estimates, cate_lower, cate_upper


# STEP 3 — Test for meaningful heterogeneity
def test_heterogeneity(
    X_df: pd.DataFrame,
    Y_diff: np.ndarray,
    forest: RegressionForest,
    cate_estimates: np.ndarray,
) -> dict:
    """
    Three tests: quintile calibration, best linear predictor, variance decomposition.
    Saves heterogeneity_test.csv.
    Returns results dict.
    """
    log.info("=" * 60)
    log.info("STEP 3 — Test for meaningful heterogeneity")
    log.info("=" * 60)

    X = X_df.values

    # train/test split
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.30, random_state=42)

    X_tr, X_te   = X[idx_tr], X[idx_te]
    Y_tr, Y_te   = Y_diff[idx_tr], Y_diff[idx_te]

    forest_cv = RegressionForest(
        n_estimators=2000,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    log.info("Fitting train-set forest for calibration test ...")
    forest_cv.fit(X_tr, Y_tr)
    cate_te = forest_cv.predict(X_te).ravel()

    # ── TEST A — Quintile calibration ──────────────────────────
    quintiles = pd.qcut(cate_te, q=5, labels=False, duplicates="drop")
    quintiles = np.array(quintiles)
    q_records = []
    for q in sorted(np.unique(quintiles[~np.isnan(quintiles)]).astype(int)):
        mask = quintiles == q
        q_records.append({
            "quintile":             int(q) + 1,
            "mean_predicted_cate":  float(cate_te[mask].mean()),
            "mean_actual_outcome":  float(Y_te[mask].mean()),
            "n":                    int(mask.sum()),
        })
    q_df = pd.DataFrame(q_records)

    actual_spread = q_df["mean_actual_outcome"].max() - q_df["mean_actual_outcome"].min()

    print("\n  TEST A — Quintile calibration (train/test split)")
    print(f"  {'Q':>3}  {'N':>5}  {'Mean predicted CATE':>20}  {'Mean actual outcome':>20}")
    for _, row in q_df.iterrows():
        print(
            f"  {int(row['quintile']):>3}  {int(row['n']):>5}  "
            f"{row['mean_predicted_cate']:>+20.4f}  "
            f"{row['mean_actual_outcome']:>+20.4f}"
        )
    print(f"\n  Spread of actual means across quintiles: {actual_spread:.4f}")
    if actual_spread > 0.10:
        quint_verdict = "Meaningful heterogeneity detected"
    elif actual_spread < 0.05:
        quint_verdict = "No meaningful heterogeneity — null is uniform"
    else:
        quint_verdict = "Weak heterogeneity — interpret with caution"
    print(f"  Verdict: {quint_verdict}")

    # ── TEST B — Best linear predictor ─────────────────────────
    slope, intercept, r, p_blp, se_blp = stats.linregress(cate_te, Y_te)
    r2_blp = r ** 2

    print(f"\n  TEST B — Best linear predictor")
    print(f"  BLP slope:   {slope:+.4f}")
    print(f"  BLP p-value: {p_blp:.4f}")
    print(f"  R²:          {r2_blp:.4f}")
    if p_blp < 0.05:
        blp_verdict = "CATE estimates have predictive signal — heterogeneity is real"
    else:
        blp_verdict = "CATE estimates do not predict outcomes — heterogeneity is noise"
    print(f"  Verdict: {blp_verdict}")

    # ── TEST C — Variance decomposition ────────────────────────
    var_cate    = float(np.var(cate_estimates))
    var_outcome = float(np.var(Y_diff))
    het_share   = var_cate / var_outcome if var_outcome > 0 else 0.0

    print(f"\n  TEST C — Variance decomposition")
    print(f"  Variance of CATE estimates: {var_cate:.6f}")
    print(f"  Variance of outcomes:       {var_outcome:.6f}")
    print(f"  Heterogeneity share:        {het_share:.4f}")
    if het_share < 0.01:
        var_verdict = (
            "Treatment effect heterogeneity explains less than 1% "
            "of outcome variance — consistent with null average effect"
        )
    elif het_share > 0.05:
        var_verdict = "Non-trivial heterogeneity detected — investigate SHAP in Step 4"
    else:
        var_verdict = "Modest heterogeneity — 1-5% of outcome variance"
    print(f"  Verdict: {var_verdict}")

    # ── Overall verdict ────────────────────────────────────────
    n_fail = sum([
        actual_spread < 0.05,
        p_blp >= 0.05,
        het_share < 0.01,
    ])
    if n_fail == 3:
        overall_verdict = (
            "No meaningful heterogeneity — the null average treatment effect "
            "is uniform across observed subgroups. The causal forest confirms "
            "the Phase 4 null result is not masking large subgroup effects."
        )
    elif n_fail >= 2:
        overall_verdict = (
            "Weak heterogeneity detected — some variation in individual treatment "
            "effects exists but explains minimal outcome variance. Subgroup "
            "differences should be interpreted cautiously."
        )
    else:
        overall_verdict = (
            "Meaningful heterogeneity detected — the null average masks genuine "
            "subgroup variation. See SHAP analysis for drivers."
        )

    print(f"\n  OVERALL VERDICT: {overall_verdict}")

    results = {
        "quintile_spread":       round(actual_spread, 6),
        "quintile_verdict":      quint_verdict,
        "blp_slope":             round(float(slope), 6),
        "blp_pvalue":            round(float(p_blp), 6),
        "blp_r2":                round(float(r2_blp), 6),
        "blp_verdict":           blp_verdict,
        "var_cate":              round(var_cate, 6),
        "var_outcome":           round(var_outcome, 6),
        "heterogeneity_share":   round(het_share, 6),
        "var_verdict":           var_verdict,
        "overall_verdict":       overall_verdict,
    }

    # save quintile table rows + summary row
    q_df["test"] = "quintile_calibration"
    summary_row  = pd.DataFrame([{
        "quintile":            None,
        "mean_predicted_cate": None,
        "mean_actual_outcome": None,
        "n":                   None,
        "test":                "summary",
        **results,
    }])
    out_df = pd.concat([q_df.assign(**{k: None for k in results}), summary_row],
                       ignore_index=True)
    out_df.to_csv(RESULTS_DIR / "heterogeneity_test.csv", index=False)
    log.info("Saved heterogeneity_test.csv")

    return results


# STEP 4 — SHAP analysis
def shap_analysis(
    forest: RegressionForest,
    X_df: pd.DataFrame,
    cate_estimates: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    """
    Computes SHAP values and produces three plots:
    shap_summary.png, shap_dependence_plots.png, cate_distribution.png.
    Returns shap_values array.
    """
    log.info("=" * 60)
    log.info("STEP 4 — SHAP analysis")
    log.info("=" * 60)

    X = X_df.values
    feature_names = list(X_df.columns)

    log.info("Computing SHAP values ...")
    explainer   = shap.TreeExplainer(forest)
    shap_values = explainer.shap_values(X)

    # ── CHART 1 — SHAP summary beeswarm ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    ax.set_title(
        "SHAP values — drivers of treatment effect heterogeneity",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel(
        "Impact on CATE estimate\n"
        "Each point = one substitution  |  Color = feature value",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_summary.png")

    # ── Top-4 features by mean |SHAP| ─────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top4_idx      = np.argsort(mean_abs_shap)[::-1][:4]
    top4_names    = [feature_names[i] for i in top4_idx]

    print("\n  Top 5 features by mean |SHAP value|:")
    print(f"  {'Rank':<5}  {'Feature':<25}  {'Mean |SHAP|':>11}  {'Direction'}")
    for rank, i in enumerate(np.argsort(mean_abs_shap)[::-1][:5], 1):
        fname = feature_names[i]
        mshap = mean_abs_shap[i]
        # direction: positive correlation between feature and SHAP
        corr_sign = np.corrcoef(X[:, i], shap_values[:, i])[0, 1]
        direction = "positive" if corr_sign >= 0 else "negative"
        print(f"  {rank:<5}  {fname:<25}  {mshap:>11.4f}  {direction}")

    # ── CHART 2 — SHAP dependence plots (2×2 grid) ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, feat_name in zip(axes, top4_names):
        feat_idx = feature_names.index(feat_name)
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X,
            feature_names=feature_names,
            interaction_index="auto",
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP dependence: {feat_name}", fontsize=11)

    fig.suptitle(
        "SHAP dependence plots — top 4 CATE heterogeneity drivers",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_dependence_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_dependence_plots.png")

    # ── CHART 3 — CATE distribution ────────────────────────────
    game_state = meta["t_game_state"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — overall histogram
    ax1.hist(cate_estimates, bins=60, color="#7F77DD", edgecolor="none", alpha=0.85)
    ax1.axvline(cate_estimates.mean(), color="black", linestyle="--", lw=1.5,
                label=f"Mean CATE = {cate_estimates.mean():+.3f}")
    ax1.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero")
    ax1.set_title(
        "Distribution of individual treatment\neffect estimates (CATEs)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xlabel("CATE (goal differential, 15-min)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.annotate(
        f"Mean CATE = {cate_estimates.mean():.3f}\n(Phase 4 ATT = {PHASE4_ATT})",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Panel 2 — KDE by game state
    colors = {"Losing": "coral", "Level": "gray", "Winning": "teal"}
    from scipy.stats import gaussian_kde
    for gs, col in colors.items():
        mask = game_state == gs
        if mask.sum() < 10:
            continue
        vals = cate_estimates[mask]
        kde  = gaussian_kde(vals, bw_method="scott")
        xs   = np.linspace(cate_estimates.min(), cate_estimates.max(), 300)
        ax2.plot(xs, kde(xs), color=col, lw=2, label=gs)
        ax2.fill_between(xs, kde(xs), alpha=0.15, color=col)

    ax2.axvline(0, color="black", linestyle="--", lw=1.5, alpha=0.6)
    ax2.set_title("CATE distribution by game state", fontsize=11, fontweight="bold")
    ax2.set_xlabel("CATE (goal differential, 15-min)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Causal forest individual treatment effect estimates\n"
        "15-minute outcome window  |  Matched pairs  |  N = {:,}".format(len(cate_estimates)),
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cate_distribution.png")

    return shap_values


# STEP 5 — Subgroup CATE summary
def subgroup_cates(
    cate_estimates: np.ndarray,
    X_df: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean CATE per subgroup and compare to Phase 4 ATT.
    """
    log.info("=" * 60)
    log.info("STEP 5 — Subgroup CATE summary")
    log.info("=" * 60)

    game_state = meta["t_game_state"].values
    league     = meta["t_league"].values
    minute     = X_df["t_minute"].values
    direction  = meta["t_sub_direction"].values

    def _group_stats(mask, label):
        vals = cate_estimates[mask]
        mean = vals.mean()
        std  = vals.std()
        n    = mask.sum()
        p4   = PHASE4_SUBGROUP_ATT.get(label, np.nan)
        consistent = (
            abs(mean - p4) < 0.03 and np.sign(mean) == np.sign(p4)
            if not np.isnan(p4) else None
        )
        flag = abs(mean) > 0.10
        return {
            "subgroup":       label,
            "n":              int(n),
            "mean_cate":      round(float(mean), 4),
            "std_cate":       round(float(std),  4),
            "phase4_att":     round(float(p4), 4) if not np.isnan(p4) else None,
            "consistent":     consistent,
            "flag_large":     flag,
        }

    rows = []

    for gs in ["Losing", "Level", "Winning"]:
        rows.append(_group_stats(game_state == gs, gs))

    for label, cond in [("Early sub < 74", minute < 74), ("Late sub >= 74", minute >= 74)]:
        rows.append(_group_stats(cond, label))

    for d in ["offensive", "defensive", "neutral"]:
        label = d.capitalize()
        rows.append(_group_stats(direction == d, label))

    for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]:
        rows.append(_group_stats(league == lg, lg))

    df = pd.DataFrame(rows)

    print("\n  Subgroup CATE vs Phase 4 ATT:")
    print(f"  {'Subgroup':<22}  {'N':>5}  {'Phase4 ATT':>11}  {'Mean CATE':>10}  {'Consistent?':>11}  {'Flag'}")
    print("  " + "-" * 75)
    for _, row in df.iterrows():
        p4_str   = f"{row['phase4_att']:+.4f}" if row["phase4_att"] is not None else "   N/A  "
        cons_str = "Yes" if row["consistent"] else ("No" if row["consistent"] is False else "—")
        flag_str = "*** FLAG" if row["flag_large"] else ""
        print(
            f"  {row['subgroup']:<22}  {row['n']:>5}  {p4_str:>11}  "
            f"{row['mean_cate']:>+10.4f}  {cons_str:>11}  {flag_str}"
        )

    return df


# Write summary
def write_summary(
    cate_estimates: np.ndarray,
    het_results: dict,
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    feature_names = list(X_df.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top5_idx      = np.argsort(mean_abs_shap)[::-1][:5]

    shap_lines = []
    for rank, i in enumerate(top5_idx, 1):
        fname = feature_names[i]
        mshap = mean_abs_shap[i]
        corr_sign = np.corrcoef(X_df.values[:, i], shap_values[:, i])[0, 1]
        direction = "Higher values push CATE up" if corr_sign >= 0 else "Higher values push CATE down"
        shap_lines.append(f"  {rank}. {fname}: mean |SHAP| = {mshap:.4f}. {direction}.")

    # overall heterogeneity verdict tag for final paragraph
    overall = het_results["overall_verdict"]
    if "No meaningful" in overall:
        uniform_phrase = "uniform — no subgroup is clearly helped by earlier substitution"
    elif "Weak" in overall:
        uniform_phrase = "largely uniform with weak, statistically unreliable variation"
    else:
        uniform_phrase = "not uniform — meaningful subgroup variation detected"

    mean_cate = cate_estimates.mean()
    diff_vs_p4 = abs(mean_cate - PHASE4_ATT)

    lines = [
        "CAUSAL FOREST RESULTS SUMMARY",
        "══════════════════════════════════════════",
        "Method: Regression Forest on matched pair",
        "        differences (econml RegressionForest)",
        f"N observations: {len(cate_estimates):,}",
        "N estimators: 2,000",
        "",
        "──────────────────────────────────────────",
        "CATE ESTIMATES",
        "──────────────────────────────────────────",
        f"Mean CATE:  {mean_cate:+.4f}",
        f"Std CATE:   {cate_estimates.std():.4f}",
        f"Range:      [{cate_estimates.min():+.4f}, {cate_estimates.max():+.4f}]",
        f"% positive: {(cate_estimates > 0).mean()*100:.1f}%",
        f"Consistent with Phase 4 ATT: {'Yes' if diff_vs_p4 < 0.05 else 'No'}",
        "",
        "──────────────────────────────────────────",
        "HETEROGENEITY TESTS",
        "──────────────────────────────────────────",
        f"Calibration test (quintile spread): {het_results['quintile_spread']:.4f}",
        f"Assessment: {het_results['quintile_verdict']}",
        "",
        f"BLP test: slope={het_results['blp_slope']:+.4f}, p={het_results['blp_pvalue']:.4f}",
        f"Assessment: {'Significant' if het_results['blp_pvalue'] < 0.05 else 'Non-significant'}",
        "",
        f"Heterogeneity share of outcome variance: {het_results['heterogeneity_share']:.4f}",
        f"Assessment: {'<1%' if het_results['heterogeneity_share'] < 0.01 else ('1-5%' if het_results['heterogeneity_share'] < 0.05 else '>5%')}",
        "",
        "OVERALL HETEROGENEITY VERDICT:",
        het_results["overall_verdict"],
        "",
        "──────────────────────────────────────────",
        "SHAP FINDINGS",
        "──────────────────────────────────────────",
        "Top 5 drivers of CATE heterogeneity:",
        *shap_lines,
        "",
        "──────────────────────────────────────────",
        "OVERALL PROJECT CONCLUSION",
        "──────────────────────────────────────────",
        (
            "Across five methods — matched pair ATT, doubly robust estimation, "
            "placebo testing, Rosenbaum sensitivity analysis, and causal forest — "
            "this study finds no average causal effect of tactical substitution "
            "timing on 15-minute post-substitution goal differential "
            f"(ATT = {PHASE4_ATT}, p = 0.125). The causal forest confirms this null "
            f"is {uniform_phrase}. The strongest predictor of post-substitution "
            "outcomes is substitute player quality (β = +0.23), though this "
            "explains only 0.43% of outcome variance — consistent with short-term "
            "match outcomes being dominated by stochastic factors beyond any single "
            "tactical decision."
        ),
        "══════════════════════════════════════════",
    ]

    out_path = RESULTS_DIR / "phase5_summary.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Saved phase5_summary.txt")
    print("\n" + "\n".join(lines))


# Orchestrator
def run_all() -> None:
    log.info("=" * 60)
    log.info("Phase 5 — Causal Forest — START")
    log.info("=" * 60)

    pairs = pd.read_parquet(PROCESSED_DIR / "matched_pairs.parquet")
    log.info("Loaded matched_pairs.parquet: %d pairs", len(pairs))

    # ── Step 1 ───────────────────────────────────────────────
    X_df, Y_diff, meta = prepare_data(pairs)

    # ── Step 2 ───────────────────────────────────────────────
    forest, cate_estimates, cate_lower, cate_upper = fit_forest(X_df, Y_diff)

    # ── Save cate_estimates.parquet ───────────────────────────
    cate_df = X_df.copy()
    cate_df["Y_diff"]        = Y_diff
    cate_df["cate_estimate"] = cate_estimates
    cate_df["cate_lower"]    = cate_lower
    cate_df["cate_upper"]    = cate_upper
    cate_df["league"]        = meta["t_league"].values
    cate_df["t_minute"]      = meta["t_minute"].values    # already in X_df but also in meta
    cate_df["t_game_state"]  = meta["t_game_state"].values
    cate_df.to_parquet(RESULTS_DIR / "cate_estimates.parquet", index=False)
    log.info("Saved cate_estimates.parquet")

    # ── Step 3 ───────────────────────────────────────────────
    het_results = test_heterogeneity(X_df, Y_diff, forest, cate_estimates)

    # ── Step 4 ───────────────────────────────────────────────
    shap_values = shap_analysis(forest, X_df, cate_estimates, meta)

    # ── Step 5 ───────────────────────────────────────────────
    subgroup_df = subgroup_cates(cate_estimates, X_df, meta)

    # ── Summary ──────────────────────────────────────────────
    write_summary(cate_estimates, het_results, shap_values, X_df, subgroup_df)

    log.info("Phase 5 — COMPLETE")
    log.info("Outputs:")
    for p in [
        "data/results/cate_estimates.parquet",
        "data/results/cate_distribution.png",
        "data/results/shap_summary.png",
        "data/results/shap_dependence_plots.png",
        "data/results/heterogeneity_test.csv",
        "data/results/phase5_summary.txt",
    ]:
        full   = ROOT / p
        status = "✓" if full.exists() else "✗ MISSING"
        log.info("  %s  %s", status, p)


if __name__ == "__main__":
    run_all()
