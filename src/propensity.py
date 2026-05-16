"""
Phase 3 — Propensity Score Estimation and Matching

Takes model_ready.parquet and produces a matched dataset of treated/control
observation pairs for ATT estimation in Phase 4.

Input:  data/processed/model_ready.parquet
Output: data/processed/analysis_sample.parquet
        data/processed/propensity_scores.parquet
        data/processed/matched_pairs.parquet
        data/processed/X_features.parquet
        data/processed/feature_scaler.pkl
        data/results/balance_report.csv
        data/results/love_plot.png
        data/results/ps_distribution.png
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# Paths and logging
# ─────────────────────────────────────────────────────────────
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
        logging.FileHandler(LOG_DIR / "propensity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
PURPLE = "#7F77DD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
GRAY   = "#888780"

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

# Propensity model features — ONLY covariates observable for both treated AND
# control observations (game state at the moment of potential substitution).
# Player-specific features (quality, fatigue, direction, position) are null for
# all control rows and thus perfectly predict treatment if included — AUC=1.0
# with zero overlap. They are excluded here and assessed in balance checks only.
CONTINUOUS_FEATURES = [
    "score_diff_at_sub",       # C1 — primary confounder
    "opponent_points_l4",      # C4 — opponent quality
    "subs_used_before",        # C7 — substitution slot context
]
BINARY_FEATURES = ["is_home", "is_winning", "is_losing"]
# League dummies built inside prepare_features(); reference = ENG-Premier League

# Balance covariates for love plot and PASS/FAIL verdict —
# only game-state features available for BOTH treated and control observations.
# player_out_minutes_l30 and log1p_player_quality are always zero for control
# rows, so their SMD is structurally ≫2.0 regardless of matching quality;
# they are reported in balance_report.csv under structural_note but excluded
# from the PASS/FAIL determination.
BALANCE_CORE = [
    "score_diff_at_sub",
    "opponent_points_l4",
    "subs_used_before",
    "is_home",
    "is_winning",
    "is_losing",
]
# Reported in CSV only, not in love plot or PASS/FAIL
BALANCE_STRUCTURAL = [
    "log1p_player_quality",
    "player_out_minutes_l30_filled",
]

CALIPER_COEF    = 0.2   # Austin (2011): 0.2 × SD(logit PS)
CALIPER_TIGHTEN = 0.15  # fallback if any post-match SMD ≥ 0.20

# Plain-English coefficient interpretations
_COEF_INTERP = {
    "score_diff_at_sub":           "winning teams substitute later — primary game-state confounder (Del Corral 2008)",
    "is_losing":                   "losing teams substitute earlier — strongest behavioral confounder",
    "is_winning":                  "winning teams substitute later — scoreline protection",
    "opponent_points_l4":          "stronger recent opponents → slightly earlier substitution",
    "log1p_player_quality":        "higher-quality substitute player → treatment moment more likely",
    "player_out_minutes_l30_filled":"higher recent load (fatigue) → substitution more likely",
    "subs_used_before":            "more subs already used → this moment is a later substitution event",
    "is_home":                     "home/away status shifts substitution timing",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _game_state_series(df: pd.DataFrame) -> pd.Series:
    """Derive Winning/Level/Losing label from is_winning / is_losing."""
    win = pd.to_numeric(df["is_winning"], errors="coerce").fillna(0).astype(int)
    los = pd.to_numeric(df["is_losing"],  errors="coerce").fillna(0).astype(int)
    return pd.Series(
        np.where(win == 1, "Winning", np.where(los == 1, "Losing", "Level")),
        index=df.index,
    )


def _primary_position(pos_str) -> str:
    """Map multi-position strings like 'FW,MF' to GK/DF/MF/FW/none."""
    if pd.isna(pos_str) or str(pos_str).strip() in ("", "none", "nan"):
        return "none"
    first = str(pos_str).split(",")[0].strip()
    _map = {
        "GK": "GK",
        "CB": "DF", "LB": "DF", "RB": "DF", "DF": "DF",
        "DM": "MF", "CM": "MF", "AM": "MF", "MF": "MF", "LM": "MF", "RM": "MF",
        "FW": "FW", "LW": "FW", "RW": "FW",
    }
    return _map.get(first, "MF")


def _compute_smd(a, b) -> float:
    """Standardized mean difference (pooled SD). Returns nan if insufficient data."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / np.sqrt(pooled_var))


def _logit_ps(ps_series: pd.Series) -> pd.Series:
    """Safe logit transform, clipped to (-10, 10)."""
    ps_clipped = ps_series.clip(1e-6, 1 - 1e-6)
    return np.log(ps_clipped / (1 - ps_clipped)).clip(-10, 10)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Filter to analysis sample
# ─────────────────────────────────────────────────────────────
def filter_sample(mr: pd.DataFrame) -> pd.DataFrame:
    """
    Remove red-card matches, injury/disciplinary subs; add flags.
    Saves data/processed/analysis_sample.parquet.
    """
    log.info("=" * 60)
    log.info("STEP 1 — Filter to analysis sample")
    log.info("=" * 60)

    n0_trt = (mr["treated"] == 1).sum()
    n0_ctl = (mr["treated"] == 0).sum()
    log.info("Input: %d treated, %d control rows", n0_trt, n0_ctl)

    # ── 1a: red card match removal ───────────────────────────
    red_match_ids = set(
        mr.loc[(mr["treated"] == 1) & (mr["is_red_card_match"] == 1), "match_id"]
    )
    n_rc_trt = mr[(mr["treated"] == 1) & mr["match_id"].isin(red_match_ids)].shape[0]
    n_rc_ctl = mr[(mr["treated"] == 0) & mr["match_id"].isin(red_match_ids)].shape[0]
    mr = mr[~mr["match_id"].isin(red_match_ids)].copy()
    log.info(
        "Red card removal: %d matches → -%d treated rows, -%d control rows",
        len(red_match_ids), n_rc_trt, n_rc_ctl,
    )

    # ── 1b: sub-type filter ───────────────────────────────────
    trt = mr[mr["treated"] == 1].copy()
    ctl = mr[mr["treated"] == 0].copy()

    type_counts = trt["sub_type"].value_counts()
    log.info("Treated sub types: %s", type_counts.to_dict())

    n_inj  = (trt["sub_type"] == "injury_proxy").sum()
    n_disc = (trt["sub_type"] == "disciplinary_proxy").sum()
    trt = trt[trt["sub_type"] == "tactical"].copy()
    log.info(
        "Removed %d injury_proxy + %d disciplinary_proxy; tactical remaining: %d",
        n_inj, n_disc, len(trt),
    )

    # ── 1c: flags ─────────────────────────────────────────────
    trt["direction_unknown"] = trt["sub_direction"].fillna("").eq("unknown").astype(int)
    ctl["direction_unknown"] = 0

    trt["extended_window_treated"] = (~trt["minute"].between(55, 85)).astype(int)
    ctl["extended_window_treated"] = 0

    df = pd.concat([trt, ctl], ignore_index=True)

    n1_trt = (df["treated"] == 1).sum()
    n1_ctl = (df["treated"] == 0).sum()
    log.info(
        "Analysis sample: %d treated, %d control (1:%.1f ratio)",
        n1_trt, n1_ctl, n1_ctl / n1_trt,
    )
    log.info(
        "  Treated retained: %.1f%% of all original treated rows",
        n1_trt / n0_trt * 100,
    )
    log.info(
        "  Extended window (treated outside 55-85): %d (%.1f%%)",
        trt["extended_window_treated"].sum(),
        trt["extended_window_treated"].mean() * 100,
    )

    df.to_parquet(PROCESSED_DIR / "analysis_sample.parquet", index=False)
    log.info("Saved analysis_sample.parquet")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Prepare features for propensity model
# ─────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Transform and one-hot encode features. Fits StandardScaler on full
    analysis sample. Saves X_features.parquet and feature_scaler.pkl.
    Returns (df_with_derived_cols, X_array, feature_name_list).
    """
    log.info("=" * 60)
    log.info("STEP 2 — Prepare features")
    log.info("=" * 60)

    df = df.copy()

    # ── A: log1p player quality (EDA Finding 3) ──────────────
    # Treated: 23 nulls → fill with 0. Control: all null → fill with 0.
    gc = pd.to_numeric(df["goal_contributions_per90"], errors="coerce").fillna(0.0)
    df["log1p_player_quality"] = np.log1p(gc)
    n_null_quality = ((df["treated"] == 1) & df["goal_contributions_per90"].isna()).sum()
    log.info(
        "log1p_player_quality: %d treated nulls filled with 0 (log1p(0)=0 preserved)",
        n_null_quality,
    )

    # ── B: derived scalar features ────────────────────────────
    df["subs_used_before"] = (df["sub_slot"] - 1).clip(0, 5).astype(float)

    # player_out_minutes_l30: control rows all null → fill with 0
    df["player_out_minutes_l30_filled"] = (
        pd.to_numeric(df["player_out_minutes_l30"], errors="coerce").fillna(0.0)
    )

    # opponent_points_l4: ~1% nulls in both groups → fill with median
    opp_median = df["opponent_points_l4"].median()
    df["opponent_points_l4"] = df["opponent_points_l4"].fillna(opp_median)
    log.info(
        "opponent_points_l4: %d nulls filled with median=%.2f",
        df["opponent_points_l4"].isna().sum(), opp_median,
    )

    # ── C: categorical features ───────────────────────────────
    # sub_direction: control rows null → fill with 'neutral'
    df["sub_direction_filled"] = df["sub_direction"].fillna("neutral")

    # player_in_position: map to GK/DF/MF/FW; control rows null → 'none'
    df["position_primary"] = df["player_in_position"].apply(_primary_position)

    # game_state label
    df["game_state"] = _game_state_series(df)

    # ── D: league dummies (reference = ENG-Premier League) ───
    lg_dummies = pd.get_dummies(df["league"], prefix="lg", dtype=float)
    lg_cols = sorted([c for c in lg_dummies.columns if "ENG" not in c])
    lg_dummies = lg_dummies[lg_cols]
    lg_dummies.columns = [
        c.replace("-", "_").replace(" ", "_") for c in lg_dummies.columns
    ]

    # ── E: standardize continuous features ───────────────────
    cont_vals = df[CONTINUOUS_FEATURES].astype(float)
    scaler = StandardScaler()
    cont_scaled = pd.DataFrame(
        scaler.fit_transform(cont_vals),
        columns=CONTINUOUS_FEATURES,
        index=df.index,
    )

    with open(PROCESSED_DIR / "feature_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("Saved feature_scaler.pkl")

    # ── F: assemble X (game-state features only) ─────────────
    # sub_direction, player_in_position, player_out_minutes_l30, and
    # log1p_player_quality are excluded from X because they are null for
    # all control rows. Including them in the propensity model creates
    # perfect separation (AUC=1.0, zero overlap) — they predict treatment
    # identity rather than the treatment assignment mechanism.
    # These features are retained in df and assessed in balance checks (Step 6).
    binary_df = df[BINARY_FEATURES].astype(float)
    X_df = pd.concat([cont_scaled, binary_df, lg_dummies], axis=1)
    feature_names = list(X_df.columns)
    X = X_df.values

    log.info("Feature matrix: %d rows × %d features", *X.shape)
    log.info("Feature list: %s", feature_names)

    X_df_save = X_df.copy()
    X_df_save["_orig_index"] = df.index
    X_df_save.to_parquet(PROCESSED_DIR / "X_features.parquet", index=False)
    log.info("Saved X_features.parquet")

    return df, X, feature_names


# ─────────────────────────────────────────────────────────────
# STEP 3 — Estimate propensity scores
# ─────────────────────────────────────────────────────────────
def estimate_propensity(
    df: pd.DataFrame, X: np.ndarray, feature_names: list
) -> tuple:
    """
    Fit logistic regression on full analysis sample.
    Adds propensity_score to df. Saves propensity_scores.parquet.
    Returns (df, model, coef_df, auc).
    """
    log.info("=" * 60)
    log.info("STEP 3 — Estimate propensity scores")
    log.info("=" * 60)

    y = df["treated"].astype(int).values

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X, y)
    log.info("Converged in %d iteration(s)", model.n_iter_[0])

    ps = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["propensity_score"] = ps

    # ── AUC ──────────────────────────────────────────────────
    auc = roc_auc_score(y, ps)
    if auc > 0.90:
        log.warning("AUC=%.4f > 0.90 — groups are very different; matching may be poor", auc)
    elif auc < 0.55:
        log.warning("AUC=%.4f < 0.55 — model has limited power", auc)
    else:
        log.info("AUC=%.4f — within target range [0.55, 0.90]", auc)

    # ── Coefficients ─────────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    log.info("Top 10 features by |coefficient|:")
    for _, row in coef_df.head(10).iterrows():
        log.info("  %-40s  %+.4f", row["feature"], row["coefficient"])

    # ── Mandatory sanity checks ───────────────────────────────
    coef_map = dict(zip(feature_names, model.coef_[0]))

    sd_coef = coef_map.get("score_diff_at_sub")
    if sd_coef is not None:
        ok = sd_coef > 0
        status = "✓ POSITIVE" if ok else "✗ ERROR — NEGATIVE"
        log.info("Sanity check 1 — score_diff_at_sub: %+.4f [%s]", sd_coef, status)
        if not ok:
            log.error("score_diff_at_sub is NEGATIVE — model learned reversed relationship")

    il_coef = coef_map.get("is_losing")
    if il_coef is not None:
        # Losing teams sub more aggressively → at any given game-moment in 55–85,
        # P(treated=1 | losing) > P(treated=1 | winning). Correct direction: POSITIVE.
        ok = il_coef > 0
        status = "✓ POSITIVE (losing → more likely to sub at any moment)" if ok else "⚠ NEGATIVE (unexpected)"
        log.info("Sanity check 2 — is_losing: %+.4f [%s]", il_coef, status)
        if not ok:
            log.warning("is_losing is NEGATIVE — may indicate low model power rather than error")

    lg_coefs = {k: v for k, v in coef_map.items() if k.startswith("lg_")}
    lg_range = max(lg_coefs.values()) - min(lg_coefs.values()) if lg_coefs else 0.0
    ok_lg = lg_range > 0.05
    log.info(
        "Sanity check 3 — league coef range: %.4f [%s]",
        lg_range, "✓" if ok_lg else "⚠ SMALL — league effects may not contribute",
    )

    # ── Plain English interpretation (top 5) ─────────────────
    log.info("Top 5 features — interpretation:")
    for i, (_, row) in enumerate(coef_df.head(5).iterrows(), 1):
        interp = _COEF_INTERP.get(row["feature"], "no interpretation recorded")
        log.info("  %d. %s (%+.3f): %s", i, row["feature"], row["coefficient"], interp)

    # ── Save ─────────────────────────────────────────────────
    ps_out = df[
        ["match_id", "league", "season", "team", "minute", "treated", "propensity_score"]
    ].copy()
    ps_out.to_parquet(PROCESSED_DIR / "propensity_scores.parquet", index=False)
    log.info("Saved propensity_scores.parquet")

    return df, model, coef_df, auc


# ─────────────────────────────────────────────────────────────
# STEP 4 — Assess common support
# ─────────────────────────────────────────────────────────────
def assess_common_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim observations outside the overlap region.
    Saves data/results/ps_distribution.png.
    Returns trimmed DataFrame.
    """
    log.info("=" * 60)
    log.info("STEP 4 — Common support")
    log.info("=" * 60)

    ps_trt = df.loc[df["treated"] == 1, "propensity_score"]
    ps_ctl = df.loc[df["treated"] == 0, "propensity_score"]

    overlap_min = max(ps_trt.min(), ps_ctl.min())
    overlap_max = min(ps_trt.max(), ps_ctl.max())
    log.info("Overlap region: [%.5f, %.5f]  width=%.5f", overlap_min, overlap_max,
             overlap_max - overlap_min)

    in_support = df["propensity_score"].between(overlap_min, overlap_max)
    n_trim_trt = ((df["treated"] == 1) & ~in_support).sum()
    n_trim_ctl = ((df["treated"] == 0) & ~in_support).sum()

    df_trim = df[in_support].copy().reset_index(drop=True)
    log.info(
        "Trimmed: %d treated, %d control outside overlap → remaining: %d treated, %d control",
        n_trim_trt, n_trim_ctl,
        (df_trim["treated"] == 1).sum(), (df_trim["treated"] == 0).sum(),
    )

    # ── Per-league overlap stats ──────────────────────────────
    log.info("Per-league overlap statistics:")
    for lg in LEAGUE_ORDER:
        ldf = df_trim[df_trim["league"] == lg]
        t = ldf.loc[ldf["treated"] == 1, "propensity_score"]
        c = ldf.loc[ldf["treated"] == 0, "propensity_score"]
        if len(t) == 0 or len(c) == 0:
            continue
        lg_overlap_min = max(t.min(), c.min())
        lg_overlap_max = min(t.max(), c.max())
        pct_in = t.between(lg_overlap_min, lg_overlap_max).mean() * 100
        log.info(
            "  %-20s treated: %.3f±%.3f  control: %.3f±%.3f  "
            "overlap width: %.3f  treated in overlap: %.1f%%",
            LEAGUE_SHORT[lg], t.mean(), t.std(), c.mean(), c.std(),
            lg_overlap_max - lg_overlap_min, pct_in,
        )

    # ── Plot ─────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)

    for ax, lg in zip(axes, LEAGUE_ORDER):
        ldf = df_trim[df_trim["league"] == lg]
        t_ps = ldf.loc[ldf["treated"] == 1, "propensity_score"]
        c_ps = ldf.loc[ldf["treated"] == 0, "propensity_score"]
        bins = np.linspace(overlap_min, overlap_max, 35)

        ax.hist(c_ps, bins=bins, density=True, alpha=0.55, color=TEAL,   label="Control")
        ax.hist(t_ps, bins=bins, density=True, alpha=0.55, color=PURPLE, label="Treated")
        ax.axvspan(overlap_min, overlap_max, alpha=0.07, color="gray", zorder=0)

        ax.set_title(LEAGUE_SHORT[lg], fontsize=10, fontweight="bold")
        ax.set_xlabel("Propensity score", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=9)

    handles = [
        mpatches.Patch(color=PURPLE, alpha=0.7, label="Treated"),
        mpatches.Patch(color=TEAL,   alpha=0.7, label="Control"),
    ]
    axes[-1].legend(handles=handles, fontsize=9)
    fig.suptitle(
        "Propensity score distributions by league (before matching)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ps_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved ps_distribution.png")

    return df_trim


# ─────────────────────────────────────────────────────────────
# STEP 5 — Nearest neighbor matching with caliper
# ─────────────────────────────────────────────────────────────
def _do_match(df: pd.DataFrame, caliper: float) -> pd.DataFrame:
    """
    Inner matching loop. Per-league KDTree nearest-neighbor without replacement.
    Returns matched_pairs DataFrame.
    """
    # Compute logit PS on trimmed sample
    df = df.copy()
    df["logit_ps"] = _logit_ps(df["propensity_score"])

    # Pre-compute game state for all rows
    df["game_state"] = _game_state_series(df)

    all_pairs: list[dict] = []

    for lg in LEAGUE_ORDER:
        ldf = df[df["league"] == lg]
        trt_l = ldf[ldf["treated"] == 1].copy()
        ctl_l = ldf[ldf["treated"] == 0].copy()

        n_trt = len(trt_l)
        n_ctl = len(ctl_l)
        if n_trt == 0 or n_ctl == 0:
            log.warning("League %s: %d treated / %d control — skipping", lg, n_trt, n_ctl)
            continue

        # Build KDTree on control logit PS
        ctrl_logit = ctl_l["logit_ps"].values.reshape(-1, 1)
        tree = KDTree(ctrl_logit)

        # Reset integer position index for O(1) iloc lookup
        ctl_l = ctl_l.reset_index(drop=True)

        # Sort treated ascending by PS (standard ordering for greedy matching)
        trt_sorted = trt_l.sort_values("propensity_score")

        used_ctrl_pos: set[int] = set()
        n_matched = 0
        n_no_cand  = 0  # no control within caliper at all
        n_exhausted = 0  # candidates exist but all already used

        for _, t_row in trt_sorted.iterrows():
            t_logit = np.array([[t_row["logit_ps"]]])
            cand_pos_arr, dist_arr = tree.query_radius(
                t_logit, r=caliper, return_distance=True
            )
            cand_pos  = cand_pos_arr[0]
            cand_dist = dist_arr[0]

            if len(cand_pos) == 0:
                n_no_cand += 1
                continue

            # Sort by distance ascending
            order = np.argsort(cand_dist)
            cand_pos  = cand_pos[order]
            cand_dist = cand_dist[order]

            matched = False
            for pos, dist in zip(cand_pos, cand_dist):
                if pos not in used_ctrl_pos:
                    c_row = ctl_l.iloc[pos]
                    used_ctrl_pos.add(pos)

                    t_pl30 = pd.to_numeric(t_row["player_out_minutes_l30"], errors="coerce")
                    t_pl30 = float(t_pl30) if pd.notna(t_pl30) else 0.0

                    all_pairs.append({
                        # ── Treated ─────────────────────────────────────
                        "t_match_id":               t_row["match_id"],
                        "t_minute":                 int(t_row["minute"]),
                        "t_team":                   t_row["team"],
                        "t_league":                 t_row["league"],
                        "t_season":                 t_row["season"],
                        "t_score_diff_at_sub":      float(t_row["score_diff_at_sub"]),
                        "t_is_home":                int(t_row["is_home"]),
                        "t_game_state":             t_row["game_state"],
                        "t_sub_direction":          t_row.get("sub_direction"),
                        "t_player_in_position":     t_row.get("position_primary"),
                        "t_log1p_player_quality":   float(t_row["log1p_player_quality"]),
                        "t_player_out_minutes_l30": t_pl30,
                        "t_subs_used_before":       int(t_row["subs_used_before"]),
                        "t_opponent_points_l4":     float(t_row["opponent_points_l4"]),
                        "t_propensity_score":       float(t_row["propensity_score"]),
                        "t_logit_ps":               float(t_row["logit_ps"]),
                        "t_goal_diff_next15":       int(t_row["goal_diff_next15"]),
                        "t_goal_diff_next30":       int(t_row["goal_diff_next30"]),
                        "t_outcome_window_truncated": int(t_row["outcome_window_truncated"]),
                        "t_direction_unknown":      int(t_row.get("direction_unknown", 0)),
                        "t_extended_window_treated": int(t_row.get("extended_window_treated", 0)),
                        # ── Control ─────────────────────────────────────
                        "c_match_id":               c_row["match_id"],
                        "c_minute":                 int(c_row["minute"]),
                        "c_team":                   c_row["team"],
                        "c_league":                 c_row["league"],
                        "c_season":                 c_row["season"],
                        "c_score_diff_at_sub":      float(c_row["score_diff_at_sub"]),
                        "c_is_home":                int(c_row["is_home"]),
                        "c_game_state":             c_row["game_state"],
                        "c_propensity_score":       float(c_row["propensity_score"]),
                        "c_logit_ps":               float(c_row["logit_ps"]),
                        "c_goal_diff_next15":       int(c_row["goal_diff_next15"]),
                        "c_goal_diff_next30":       int(c_row["goal_diff_next30"]),
                        "c_outcome_window_truncated": int(c_row["outcome_window_truncated"]),
                        # ── Pair-level ───────────────────────────────────
                        "league":         lg,
                        "season":         t_row["season"],
                        "ps_distance":    float(dist),
                        "within_caliper": True,
                    })
                    n_matched += 1
                    matched = True
                    break

            if not matched:
                n_exhausted += 1

        match_rate = n_matched / n_trt * 100
        log.info(
            "  %-20s  %4d treated → %4d matched (%.1f%%)  "
            "unmatched: %d no-candidate, %d controls-exhausted  "
            "mean PS dist: %.4f",
            LEAGUE_SHORT[lg], n_trt, n_matched, match_rate,
            n_no_cand, n_exhausted,
            np.mean([p["ps_distance"] for p in all_pairs if p["league"] == lg]) if n_matched else 0,
        )
        if match_rate < 60:
            log.warning("  %s match rate %.1f%% < 60%%", LEAGUE_SHORT[lg], match_rate)

    if not all_pairs:
        raise RuntimeError("No pairs matched — check caliper and common support.")

    return pd.DataFrame(all_pairs)


def match_observations(df: pd.DataFrame) -> tuple:
    """
    Compute caliper, run per-league KDTree matching.
    Saves matched_pairs.parquet. Returns (pairs_df, caliper_used).
    """
    log.info("=" * 60)
    log.info("STEP 5 — Nearest neighbor matching with caliper")
    log.info("=" * 60)

    logit = _logit_ps(df["propensity_score"])
    caliper = CALIPER_COEF * logit.std()
    log.info(
        "Caliper = 0.2 × std(logit_ps) = 0.2 × %.4f = %.4f  [Austin 2011]",
        logit.std(), caliper,
    )

    pairs = _do_match(df, caliper)

    n_matched = len(pairs)
    n_treated = (df["treated"] == 1).sum()
    overall_rate = n_matched / n_treated * 100
    mean_dist    = pairs["ps_distance"].mean()

    log.info(
        "Matching complete: %d pairs from %d treated (%.1f%% match rate)  "
        "mean PS distance: %.4f",
        n_matched, n_treated, overall_rate, mean_dist,
    )

    pairs.to_parquet(PROCESSED_DIR / "matched_pairs.parquet", index=False)
    log.info("Saved matched_pairs.parquet")

    return pairs, caliper


# ─────────────────────────────────────────────────────────────
# STEP 6 — Balance assessment and love plot
# ─────────────────────────────────────────────────────────────
def assess_balance(
    df: pd.DataFrame, pairs: pd.DataFrame, caliper: float
) -> tuple:
    """
    Compute pre- and post-match SMD for all covariates.
    Saves balance_report.csv and love_plot.png.
    Returns (balance_df, assessment_str).
    """
    log.info("=" * 60)
    log.info("STEP 6 — Balance assessment")
    log.info("=" * 60)

    pre_trt = df[df["treated"] == 1]
    pre_ctl = df[df["treated"] == 0]

    # Reconstruct post-match treated / control from pairs
    # Treated: join on match_id + minute + team
    t_keys = pairs[["t_match_id", "t_minute", "t_team"]].rename(
        columns={"t_match_id": "match_id", "t_minute": "minute", "t_team": "team"}
    ).drop_duplicates()
    c_keys = pairs[["c_match_id", "c_minute", "c_team"]].rename(
        columns={"c_match_id": "match_id", "c_minute": "minute", "c_team": "team"}
    ).drop_duplicates()

    post_trt = df.merge(t_keys.assign(_keep=1), on=["match_id", "minute", "team"], how="inner")
    post_ctl = df.merge(c_keys.assign(_keep=1), on=["match_id", "minute", "team"], how="inner")

    records: list[dict] = []

    # ── Game-state confounders (used for PASS/FAIL verdict) ───
    log.info("Game-state covariate balance (determines PASS/FAIL):")
    for cov in BALANCE_CORE:
        smd_b = _compute_smd(pre_trt[cov].astype(float),  pre_ctl[cov].astype(float))
        smd_a = _compute_smd(post_trt[cov].astype(float), post_ctl[cov].astype(float))
        balanced = bool(smd_a < 0.10) if not np.isnan(smd_a) else False
        if np.isnan(smd_a):
            flag = ""
        elif smd_a >= 0.20:
            flag = "ERROR"
        elif smd_a >= 0.10:
            flag = "WARNING"
        else:
            flag = ""
        records.append({
            "covariate": cov,
            "smd_before": round(smd_b, 4) if not np.isnan(smd_b) else None,
            "smd_after":  round(smd_a, 4) if not np.isnan(smd_a) else None,
            "balanced": balanced,
            "flag": flag,
            "structural_note": "",
        })
        marker = {"ERROR": "✗ ERROR", "WARNING": "⚠ WARNING"}.get(flag, "✓")
        log.info(
            "  %-35s  before=%.3f  after=%.3f  %s",
            cov,
            smd_b if not np.isnan(smd_b) else -1,
            smd_a if not np.isnan(smd_a) else -1,
            marker,
        )

    # ── Structural covariates (reported only, excluded from verdict) ──
    # These are always null for control rows (filled with 0), so their
    # SMD is structurally high regardless of matching quality — not a
    # matching failure, but an inherent property of the control pool design.
    log.info("Structural covariates (reported in CSV, excluded from PASS/FAIL):")
    for cov in BALANCE_STRUCTURAL:
        smd_b = _compute_smd(pre_trt[cov].astype(float),  pre_ctl[cov].astype(float))
        smd_a = _compute_smd(post_trt[cov].astype(float), post_ctl[cov].astype(float))
        note = "structural: null for all control rows; SMD is not a matching-quality indicator"
        records.append({
            "covariate": cov,
            "smd_before": round(smd_b, 4) if not np.isnan(smd_b) else None,
            "smd_after":  round(smd_a, 4) if not np.isnan(smd_a) else None,
            "balanced": False,
            "flag": "STRUCTURAL",
            "structural_note": note,
        })
        log.info(
            "  %-35s  before=%.3f  after=%.3f  STRUCTURAL (excluded from verdict)",
            cov,
            smd_b if not np.isnan(smd_b) else -1,
            smd_a if not np.isnan(smd_a) else -1,
        )

    bal_df = pd.DataFrame(records)
    bal_df.to_csv(RESULTS_DIR / "balance_report.csv", index=False)
    log.info("Saved balance_report.csv")

    # ── Overall assessment (game-state covariates only) ───────
    core_df = bal_df[bal_df["flag"] != "STRUCTURAL"]
    n_err  = (core_df["flag"] == "ERROR").sum()
    n_warn = (core_df["flag"] == "WARNING").sum()
    n_good = (core_df["flag"] == "").sum()
    worst  = core_df.loc[core_df["smd_after"].idxmax()]
    assessment = "FAIL" if n_err > 0 else "MARGINAL" if n_warn > 0 else "PASS"

    log.info(
        "Balance verdict (game-state covariates): %d PASS / %d WARNING / %d ERROR  →  %s",
        n_good, n_warn, n_err, assessment,
    )
    log.info(
        "Worst game-state covariate: %s  post-match SMD=%.4f",
        worst["covariate"], worst["smd_after"],
    )
    if n_err > 0:
        log.error(
            "BALANCE FAILED — tighten caliper from %.3f to %.3f×SD and re-run matching",
            caliper, CALIPER_TIGHTEN,
        )

    _plot_love(bal_df, len(pairs))

    return bal_df, assessment


def _plot_love(bal_df: pd.DataFrame, n_matched: int) -> None:
    """Generate the love plot (game-state confounders only)."""
    sorted_df = (
        bal_df[bal_df["flag"] != "STRUCTURAL"]
        .sort_values("smd_before", ascending=False)
        .reset_index(drop=True)
    )
    labels = (
        sorted_df["covariate"]
        .str.replace("_filled", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    max_x = max(sorted_df["smd_before"].max(), sorted_df["smd_after"].max()) + 0.05

    # Background regions
    ax.axvspan(0,    0.10,  alpha=0.10, color="green",  zorder=0, label="_nolegend_")
    ax.axvspan(0.10, 0.20,  alpha=0.10, color="yellow", zorder=0, label="_nolegend_")
    ax.axvspan(0.20, max_x, alpha=0.10, color="red",    zorder=0, label="_nolegend_")

    y = np.arange(len(sorted_df))

    ax.scatter(
        sorted_df["smd_before"], y,
        color=CORAL, marker="o", facecolors="none",
        s=90, linewidths=2, zorder=3, label="Pre-match",
    )
    ax.scatter(
        sorted_df["smd_after"], y,
        color=TEAL, marker="o", s=90, zorder=4, label="Post-match",
    )

    ax.axvline(0.10, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(0.20, color="black", linestyle=":",  linewidth=1.2, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Standardized Mean Difference (SMD)", fontsize=12)
    ax.set_xlim(0, max_x)
    ax.set_title(
        "Covariate balance before and after propensity score matching",
        fontsize=13, fontweight="bold",
    )
    ax.text(
        0.98, 0.02, f"n = {n_matched:,} matched pairs",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color=GRAY,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "love_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved love_plot.png")


# ─────────────────────────────────────────────────────────────
# Validation summary (Section 4 of spec)
# ─────────────────────────────────────────────────────────────
def _print_validation(
    mr_orig: pd.DataFrame,
    df_sample: pd.DataFrame,
    pairs: pd.DataFrame,
    bal_df: pd.DataFrame,
    assessment: str,
    caliper: float,
    coef_df: pd.DataFrame,
    auc: float,
) -> None:
    trt = df_sample[df_sample["treated"] == 1]
    ctl = df_sample[df_sample["treated"] == 0]

    print("\n" + "=" * 60)
    print("PHASE 3 VALIDATION SUMMARY")
    print("=" * 60)

    # 1. Sample flow
    n_orig_tact = (mr_orig[mr_orig["treated"] == 1]["sub_type"] == "tactical").sum()
    n_rc = (mr_orig[(mr_orig["treated"] == 1) & (mr_orig["is_red_card_match"] == 1)]).shape[0]
    n_inj  = (mr_orig[mr_orig["treated"] == 1]["sub_type"] == "injury_proxy").sum()
    n_disc = (mr_orig[mr_orig["treated"] == 1]["sub_type"] == "disciplinary_proxy").sum()
    n_final_trt = (trt.shape[0])
    n_final_ctl = (ctl.shape[0])

    print("\n1. SAMPLE FLOW")
    print(f"   Original tactical treated rows:     {n_orig_tact:>7,}")
    print(f"   After red card match removal:       {n_orig_tact - n_rc:>7,}  (-{n_rc:,})")
    print(f"   After injury/disciplinary removal:  {n_final_trt:>7,}  "
          f"(-{n_inj + n_disc:,})")
    print(f"   Control rows in analysis sample:    {n_final_ctl:>7,}")
    print(f"   Treated/control ratio: 1:{n_final_ctl/n_final_trt:.1f}")

    # 2. Propensity model
    sd_coef = coef_df.loc[coef_df["feature"] == "score_diff_at_sub", "coefficient"].values
    il_coef = coef_df.loc[coef_df["feature"] == "is_losing", "coefficient"].values
    sd_val = sd_coef[0] if len(sd_coef) else float("nan")
    il_val = il_coef[0] if len(il_coef) else float("nan")

    print(f"\n2. PROPENSITY MODEL")
    print(f"   AUC (training): {auc:.4f}")
    print(f"   score_diff_at_sub coef: {sd_val:+.4f}  {'✓' if sd_val > 0 else '✗'}")
    print(f"   is_losing coef:         {il_val:+.4f}  {'✓' if il_val > 0 else '✗'}  (POSITIVE expected: losing → more likely to sub)")
    print(f"   Top 5 features by |coef|:")
    for _, row in coef_df.head(5).iterrows():
        print(f"     {row['feature']:40s}  {row['coefficient']:+.4f}")

    # 3. Common support
    ps_trt = trt["propensity_score"]
    ps_ctl = ctl["propensity_score"]
    ov_min = max(ps_trt.min(), ps_ctl.min())
    ov_max = min(ps_trt.max(), ps_ctl.max())
    n_trim_trt = (~ps_trt.between(ov_min, ov_max)).sum()
    n_trim_ctl = (~ps_ctl.between(ov_min, ov_max)).sum()

    print(f"\n3. COMMON SUPPORT")
    print(f"   Overlap region: [{ov_min:.4f}, {ov_max:.4f}]")
    print(f"   Trimmed from treated: {n_trim_trt}")
    print(f"   Trimmed from control: {n_trim_ctl}")

    # 4. Matching results table
    print(f"\n4. MATCHING RESULTS")
    print(f"   {'League':<22} {'Treated':>8} {'Matched':>8} {'Rate%':>7} {'Mean PS dist':>13}")
    print(f"   {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*13}")
    total_trt = total_matched = 0
    for lg in LEAGUE_ORDER:
        n_lg_trt = (trt["league"] == lg).sum()
        n_lg_mat = (pairs["league"] == lg).sum()   # .sum() not .shape[0]
        rate = n_lg_mat / n_lg_trt * 100 if n_lg_trt > 0 else 0
        mean_d = pairs.loc[pairs["league"] == lg, "ps_distance"].mean() if n_lg_mat > 0 else float("nan")
        flag = "  ⚠ <60%" if rate < 60 else ""
        print(f"   {LEAGUE_SHORT[lg]:<22} {n_lg_trt:>8,} {n_lg_mat:>8,} {rate:>6.1f}% {mean_d:>12.4f}{flag}")
        total_trt += n_lg_trt
        total_matched += n_lg_mat
    overall_rate = total_matched / total_trt * 100 if total_trt > 0 else 0
    print(f"   {'TOTAL':<22} {total_trt:>8,} {total_matched:>8,} {overall_rate:>6.1f}%  {pairs['ps_distance'].mean():>12.4f}")
    print(f"   Caliper used: {caliper:.4f} (0.2 × SD[logit PS])")

    # 5. Balance (game-state confounders only)
    core_bal = bal_df[bal_df["flag"] != "STRUCTURAL"]
    n_good = (core_bal["flag"] == "").sum()
    n_warn = (core_bal["flag"] == "WARNING").sum()
    n_err  = (core_bal["flag"] == "ERROR").sum()
    worst  = core_bal.loc[core_bal["smd_after"].idxmax()]

    print(f"\n5. BALANCE RESULTS (game-state confounders, n={len(core_bal)} covariates)")
    print(f"   Balanced (SMD<0.10):    {n_good} / {len(core_bal)}")
    print(f"   Marginal (0.10-0.20):   {n_warn} / {len(core_bal)}")
    print(f"   Poor (SMD≥0.20):        {n_err} / {len(core_bal)}")
    print(f"   Worst covariate: {worst['covariate']} (post-match SMD={worst['smd_after']:.4f})")
    print(f"   Overall assessment: {assessment}")
    print(f"   Note: player_out_minutes_l30 and log1p_player_quality have structural")
    print(f"         SMD>2.0 because control rows have no substitution player (by design).")

    # 6. Naive outcome on matched sample
    matched_trt = pairs[pairs["t_outcome_window_truncated"] == 0]
    matched_ctl = pairs[pairs["c_outcome_window_truncated"] == 0]
    mean_t = matched_trt["t_goal_diff_next30"].mean()
    mean_c = pairs.loc[pairs["t_outcome_window_truncated"] == 0, "c_goal_diff_next30"].mean()
    raw_diff = mean_t - mean_c

    print(f"\n6. NAIVE OUTCOME ON MATCHED SAMPLE (non-truncated pairs only)")
    print(f"   Treated mean goal_diff_next30: {mean_t:+.4f}")
    print(f"   Control mean goal_diff_next30: {mean_c:+.4f}")
    print(f"   Raw difference (T-C):          {raw_diff:+.4f}")
    print(f"   NOTE: This is NOT the ATT — standard errors and formal")
    print(f"         estimation happen in Phase 4.")

    # 7. Game state distribution
    gs_pre  = trt["game_state"].value_counts(normalize=True).mul(100).round(1)
    gs_post = (
        pairs["t_game_state"].value_counts(normalize=True).mul(100).round(1)
    )
    print(f"\n7. GAME STATE DISTRIBUTION (treated)")
    print(f"   {'State':<10} {'Pre-match':>12} {'Post-match':>12}")
    for gs in ["Losing", "Level", "Winning"]:
        pre_pct  = gs_pre.get(gs, 0.0)
        post_pct = gs_post.get(gs, 0.0)
        print(f"   {gs:<10} {pre_pct:>11.1f}%  {post_pct:>11.1f}%")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────
def run_all() -> tuple:
    log.info("=" * 60)
    log.info("Phase 3 — Propensity Score Matching — START")
    log.info("=" * 60)

    mr = pd.read_parquet(PROCESSED_DIR / "model_ready.parquet")
    log.info("Loaded model_ready.parquet: %d rows", len(mr))

    df            = filter_sample(mr)
    df, X, fnames = prepare_features(df)
    df, model, coef_df, auc = estimate_propensity(df, X, fnames)
    df            = assess_common_support(df)
    pairs, caliper = match_observations(df)
    bal_df, assessment = assess_balance(df, pairs, caliper)

    # ── Re-run with tighter caliper if GAME-STATE balance fails ─
    # Structural covariates (player quality, fatigue) always imbalanced by
    # design and do not trigger re-matching.
    if assessment == "FAIL":
        log.warning(
            "Game-state balance FAILED. Re-running with caliper_coef=%.2f", CALIPER_TIGHTEN
        )
        logit = _logit_ps(df["propensity_score"])
        caliper_tight = CALIPER_TIGHTEN * logit.std()
        log.info("Tight caliper: %.4f", caliper_tight)
        pairs = _do_match(df, caliper_tight)
        caliper = caliper_tight
        pairs.to_parquet(PROCESSED_DIR / "matched_pairs.parquet", index=False)
        log.info("Saved matched_pairs.parquet (re-run, tight caliper)")
        bal_df, assessment = assess_balance(df, pairs, caliper)

    _print_validation(mr, df, pairs, bal_df, assessment, caliper, coef_df, auc)

    log.info("Phase 3 — COMPLETE")
    log.info("Outputs:")
    for p in [
        "data/processed/analysis_sample.parquet",
        "data/processed/propensity_scores.parquet",
        "data/processed/matched_pairs.parquet",
        "data/processed/X_features.parquet",
        "data/processed/feature_scaler.pkl",
        "data/results/balance_report.csv",
        "data/results/love_plot.png",
        "data/results/ps_distribution.png",
    ]:
        full = ROOT / p
        exists = "✓" if full.exists() else "✗ MISSING"
        log.info("  %s  %s", exists, p)

    return df, pairs, bal_df


if __name__ == "__main__":
    run_all()
