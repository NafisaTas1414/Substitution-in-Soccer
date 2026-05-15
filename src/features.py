"""
Phase 2 — Feature Engineering
Builds model_ready.parquet from raw FBref parquet files.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INTER_DIR = PROCESSED_DIR / "intermediate"

for _d in [PROCESSED_DIR, INTER_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_MIN_START = 55
CONTROL_MIN_END = 85
FATIGUE_DAYS = 30
OPP_FORM_MATCHES = 4

# Position hierarchy for sub direction: GK < DF < MF < FW
_POS_RANK = {
    "GK": 0,
    "CB": 1, "LB": 1, "RB": 1, "DF": 1,
    "DM": 2, "CM": 2, "AM": 2, "MF": 2, "LM": 2, "RM": 2,
    "FW": 3, "LW": 3, "RW": 3,
}


def _pos_rank(pos_str):
    if pd.isna(pos_str):
        return np.nan
    primary = str(pos_str).split(",")[0].strip()
    return _POS_RANK.get(primary, np.nan)


def _save_inter(df: pd.DataFrame, name: str) -> None:
    path = INTER_DIR / name
    df.to_parquet(path, index=False)
    log.info("  saved %-45s %7d rows", name, len(df))


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------

def _load_raw():
    log.info("Loading raw parquet files...")
    subs    = pd.read_parquet(RAW_DIR / "fbref_substitutions.parquet")
    events  = pd.read_parquet(RAW_DIR / "fbref_match_events.parquet")
    matches = pd.read_parquet(RAW_DIR / "fbref_matches.parquet")
    pm      = pd.read_parquet(RAW_DIR / "fbref_player_minutes.parquet")
    pss     = pd.read_parquet(RAW_DIR / "fbref_player_season_stats.parquet")
    tms     = pd.read_parquet(RAW_DIR / "fbref_team_match_stats.parquet")

    n_before = len(subs)
    subs = subs.dropna(subset=["minute"]).copy()
    subs["minute"] = subs["minute"].astype(int)
    log.info("  subs: %d rows (%d dropped — missing minute)", len(subs), n_before - len(subs))

    matches["date"] = pd.to_datetime(matches["date"])
    tms["date"]     = pd.to_datetime(tms["date"])

    return subs, events, matches, pm, pss, tms


# ---------------------------------------------------------------------------
# Goal event helpers
# ---------------------------------------------------------------------------

def _build_goal_events(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Return goal events with a 'scoring_team' column.
    In soccerdata FBref, for both 'goal' and 'own_goal', the 'team' column
    identifies the BENEFITING team (who gained the goal), so no inversion needed.
    """
    goals = events[events["event_type"].isin(["goal", "own_goal"])].copy()
    goals["minute"] = pd.to_numeric(goals["minute"], errors="coerce").fillna(0).astype(int)
    goals["scoring_team"] = goals["team"]
    return goals[["match_id", "scoring_team", "minute"]].copy()


def _goals_in_window(
    df: pd.DataFrame,
    goals: pd.DataFrame,
    team_col: str,
    minute_col: str,
    end_offset: int,
) -> tuple:
    """
    For each row in df, count goals scored FOR and AGAINST the team in
    (minute_col, minute_col + end_offset].
    Returns (goals_for array, goals_against array) each of length len(df).
    """
    left = df[["match_id", team_col, minute_col]].copy()
    left["_rid"] = np.arange(len(left))

    merged = left.merge(
        goals.rename(columns={"minute": "gmin", "scoring_team": "gteam"}),
        on="match_id",
        how="left",
    )

    has_goal = merged["gteam"].notna()
    in_win   = has_goal & (merged["gmin"] > merged[minute_col]) & (merged["gmin"] <= merged[minute_col] + end_offset)
    merged["gf"] = (in_win & (merged["gteam"] == merged[team_col])).astype(int)
    merged["ga"] = (in_win & (merged["gteam"] != merged[team_col])).astype(int)

    agg = merged.groupby("_rid")[["gf", "ga"]].sum()
    idx = np.arange(len(df))
    return agg["gf"].reindex(idx, fill_value=0).values, agg["ga"].reindex(idx, fill_value=0).values


def _score_at_minute(
    df: pd.DataFrame,
    goals: pd.DataFrame,
    team_col: str,
    minute_col: str,
) -> tuple:
    """
    For each row, count goals FOR and AGAINST the team with goal_minute <= minute_col.
    """
    left = df[["match_id", team_col, minute_col]].copy()
    left["_rid"] = np.arange(len(left))

    merged = left.merge(
        goals.rename(columns={"minute": "gmin", "scoring_team": "gteam"}),
        on="match_id",
        how="left",
    )

    has_goal = merged["gteam"].notna()
    before   = has_goal & (merged["gmin"] <= merged[minute_col])
    merged["gf"] = (before & (merged["gteam"] == merged[team_col])).astype(int)
    merged["ga"] = (before & (merged["gteam"] != merged[team_col])).astype(int)

    agg = merged.groupby("_rid")[["gf", "ga"]].sum()
    idx = np.arange(len(df))
    return agg["gf"].reindex(idx, fill_value=0).values, agg["ga"].reindex(idx, fill_value=0).values


# ---------------------------------------------------------------------------
# Opponent quality helper (reused for treated and controls)
# ---------------------------------------------------------------------------

def _opponent_quality(pairs: pd.DataFrame, tms: pd.DataFrame) -> pd.DataFrame:
    """
    pairs: DataFrame with columns [match_id, opponent, date, _pid]
           where _pid is a unique row identifier.
    Returns DataFrame indexed by _pid with opponent_points_l4,
    opponent_cumulative_points, opponent_form_flag.
    """
    tms_q = tms[["team", "date", "points"]].rename(
        columns={"date": "tms_date", "points": "opp_pts"}
    )

    joined = pairs.merge(tms_q, left_on="opponent", right_on="team", how="left")
    joined = joined[joined["tms_date"] < joined["date"]]
    joined = joined.sort_values(["_pid", "tms_date"], ascending=[True, False])
    joined["rank"] = joined.groupby("_pid").cumcount() + 1

    last_n = joined[joined["rank"] <= OPP_FORM_MATCHES]
    agg_n  = last_n.groupby("_pid")["opp_pts"].agg(
        opponent_points_l4="sum",
        _n_matches="count",
    )
    agg_all = joined.groupby("_pid")["opp_pts"].sum().rename("opponent_cumulative_points")

    out = agg_n.join(agg_all, how="outer")
    out["opponent_form_flag"] = (out["_n_matches"].fillna(0) < OPP_FORM_MATCHES).astype(int)
    out = out.drop(columns=["_n_matches"])
    return out


# ---------------------------------------------------------------------------
# Step 1 — Score at substitution
# ---------------------------------------------------------------------------

def build_step_1(subs: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 1 — Score at substitution")
    df = subs.copy()

    parsed = df["score_at_sub"].str.split(":", expand=True)
    df["_hs"] = pd.to_numeric(parsed[0], errors="coerce")
    df["_as"] = pd.to_numeric(parsed[1] if 1 in parsed.columns else pd.Series(dtype=float), errors="coerce")
    df["score_parse_error"] = df["_hs"].isna() | df["_as"].isna()

    df = df.merge(matches[["match_id", "home_team", "away_team", "date"]], on="match_id", how="left")
    df["is_home"] = (df["team"] == df["home_team"]).astype(int)

    df["goals_for_at_sub"]     = np.where(df["is_home"] == 1, df["_hs"], df["_as"])
    df["goals_against_at_sub"] = np.where(df["is_home"] == 1, df["_as"], df["_hs"])
    df["score_diff_at_sub"]    = df["goals_for_at_sub"] - df["goals_against_at_sub"]
    df["is_winning"] = (df["score_diff_at_sub"] > 0).astype("Int8")
    df["is_level"]   = (df["score_diff_at_sub"] == 0).astype("Int8")
    df["is_losing"]  = (df["score_diff_at_sub"] < 0).astype("Int8")

    for col in ["goals_for_at_sub", "goals_against_at_sub", "score_diff_at_sub",
                "is_winning", "is_level", "is_losing"]:
        df.loc[df["score_parse_error"], col] = pd.NA

    df = df.drop(columns=["_hs", "_as"])
    log.info("  score_parse_error: %d rows (%.1f%%)",
             df["score_parse_error"].sum(), df["score_parse_error"].mean() * 100)
    _save_inter(df, "subs_with_score.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Sub classification
# ---------------------------------------------------------------------------

def build_step_2(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 2 — Sub classification")
    df = df.copy()
    df["_rid"] = np.arange(len(df))

    # Yellow card for player_out before the sub minute
    yc = events[events["event_type"] == "yellow_card"][["match_id", "player", "minute"]].copy()
    yc["minute"] = pd.to_numeric(yc["minute"], errors="coerce")

    yc_join = df[["_rid", "match_id", "player_out", "minute"]].merge(
        yc.rename(columns={"player": "player_out", "minute": "yc_min"}),
        on=["match_id", "player_out"],
        how="left",
    )
    yc_join["_yc_before"] = yc_join["yc_min"] < yc_join["minute"]
    yc_flag = yc_join.groupby("_rid")["_yc_before"].any()
    df["is_yellow_card_sub"] = yc_flag.reindex(df["_rid"], fill_value=False).values.astype(int)

    # Red card (any player) before the sub minute
    rc = events[events["event_type"].isin(["red_card", "yellow_red_card"])][
        ["match_id", "minute"]
    ].copy()
    rc["minute"] = pd.to_numeric(rc["minute"], errors="coerce")

    rc_join = df[["_rid", "match_id", "minute"]].merge(
        rc.rename(columns={"minute": "rc_min"}),
        on="match_id",
        how="left",
    )
    rc_join["_rc_before"] = rc_join["rc_min"] < rc_join["minute"]
    rc_flag = rc_join.groupby("_rid")["_rc_before"].any()
    df["is_red_card_match"] = rc_flag.reindex(df["_rid"], fill_value=False).values.astype(int)

    def _classify(row):
        if row["sub_slot"] == 1 and row["minute"] < 60:
            return "injury_proxy"
        if row["is_yellow_card_sub"] == 1:
            return "disciplinary_proxy"
        return "tactical"

    df["sub_type"] = df.apply(_classify, axis=1)
    df = df.drop(columns=["_rid"])

    log.info("  sub_type:\n%s", df["sub_type"].value_counts().to_string())
    _save_inter(df, "subs_classified.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Sub direction
# ---------------------------------------------------------------------------

def build_step_3(df: pd.DataFrame, pm: pd.DataFrame, pss: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 3 — Sub direction")
    df = df.copy()

    # Position lookup: per-match preferred, season-stats fallback
    pm_pos = pm[["match_id", "player", "position"]].drop_duplicates(subset=["match_id", "player"])
    pss_pos = pss[["player", "season", "position"]].drop_duplicates(subset=["player", "season"])

    for role in ["player_in", "player_out"]:
        df = df.merge(
            pm_pos.rename(columns={"player": role, "position": f"_{role}_pos_pm"}),
            on=["match_id", role],
            how="left",
        )
        df = df.merge(
            pss_pos.rename(columns={"player": role, "position": f"_{role}_pos_pss"}),
            on=[role, "season"],
            how="left",
        )
        df[f"{role}_position"] = df[f"_{role}_pos_pm"].combine_first(df[f"_{role}_pos_pss"])
        df = df.drop(columns=[f"_{role}_pos_pm", f"_{role}_pos_pss"])

    df["_in_rank"]  = df["player_in_position"].apply(_pos_rank)
    df["_out_rank"] = df["player_out_position"].apply(_pos_rank)

    def _direction(row):
        if pd.isna(row["_in_rank"]) or pd.isna(row["_out_rank"]):
            return "unknown"
        if row["_in_rank"] > row["_out_rank"]:
            return "offensive"
        if row["_in_rank"] < row["_out_rank"]:
            return "defensive"
        return "neutral"

    df["sub_direction"] = df.apply(_direction, axis=1)
    df = df.drop(columns=["_in_rank", "_out_rank"])

    log.info("  sub_direction:\n%s", df["sub_direction"].value_counts().to_string())
    log.info("  player_in_position null: %.1f%%", df["player_in_position"].isna().mean() * 100)
    _save_inter(df, "subs_with_direction.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Fatigue (rolling 30-day minutes for player_out)
# ---------------------------------------------------------------------------

def build_step_4(df: pd.DataFrame, pm: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 4 — Fatigue")
    df = df.copy()

    if "date" not in df.columns:
        df = df.merge(matches[["match_id", "date"]], on="match_id", how="left")
    df["date"] = pd.to_datetime(df["date"])

    pm_dated = pm[["match_id", "player", "minutes_played"]].merge(
        matches[["match_id", "date"]], on="match_id", how="left"
    ).dropna(subset=["date"])
    pm_dated["date"] = pd.to_datetime(pm_dated["date"])

    df["_rid"] = np.arange(len(df))
    left = df[["_rid", "player_out", "date"]].rename(columns={"player_out": "player", "date": "sub_date"})

    fat = left.merge(
        pm_dated[["player", "date", "minutes_played"]].rename(columns={"date": "pm_date"}),
        on="player",
        how="left",
    )
    fat = fat[
        (fat["pm_date"] >= fat["sub_date"] - pd.Timedelta(days=FATIGUE_DAYS)) &
        (fat["pm_date"] < fat["sub_date"])
    ]
    fat_agg = fat.groupby("_rid")["minutes_played"].sum()

    df["player_out_minutes_l30"] = fat_agg.reindex(df["_rid"], fill_value=0).values
    df["player_out_minutes_l30_flag"] = df["date"].isna().astype(int)
    df = df.drop(columns=["_rid"])

    log.info("  player_out_minutes_l30: mean=%.0f  median=%.0f  max=%.0f",
             df["player_out_minutes_l30"].mean(),
             df["player_out_minutes_l30"].median(),
             df["player_out_minutes_l30"].max())
    _save_inter(df, "subs_with_fatigue.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Player quality (player_in season stats)
# ---------------------------------------------------------------------------

def build_step_5(df: pd.DataFrame, pss: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 5 — Player quality")
    df = df.copy()

    q = pss.copy()
    q["minutes_played_season"] = pd.to_numeric(q["minutes_played_season"], errors="coerce")
    q["goals_season"]          = pd.to_numeric(q["goals_season"], errors="coerce")
    q["assists_season"]        = pd.to_numeric(q["assists_season"], errors="coerce")

    denom = q["minutes_played_season"].replace(0, np.nan)
    q["goals_per90"]              = q["goals_season"]   / denom * 90
    q["assists_per90"]            = q["assists_season"]  / denom * 90
    q["goal_contributions_per90"] = q["goals_per90"].fillna(0) + q["assists_per90"].fillna(0)

    keep = ["player", "season", "position", "player_quality_flag",
            "goals_per90", "assists_per90", "goal_contributions_per90"]
    q = q[keep].drop_duplicates(subset=["player", "season"])

    df = df.merge(
        q.rename(columns={
            "player":              "player_in",
            "position":            "player_in_quality_position",
            "player_quality_flag": "player_in_quality_flag",
        }),
        on=["player_in", "season"],
        how="left",
    )

    log.info("  player_in quality match rate: %.1f%%", (1 - df["goals_per90"].isna().mean()) * 100)
    _save_inter(df, "subs_with_quality.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Opponent quality
# ---------------------------------------------------------------------------

def build_step_6(df: pd.DataFrame, matches: pd.DataFrame, tms: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 6 — Opponent quality")
    df = df.copy()

    if "date" not in df.columns:
        df = df.merge(matches[["match_id", "date"]], on="match_id", how="left")
        df["date"] = pd.to_datetime(df["date"])
    if "home_team" not in df.columns:
        df = df.merge(matches[["match_id", "home_team", "away_team"]], on="match_id", how="left")

    df["opponent"] = np.where(df["team"] == df["home_team"], df["away_team"], df["home_team"])

    tms_q = tms.copy()
    tms_q["points"] = tms_q["result"].map({"W": 3, "D": 1, "L": 0})

    # Deduplicate to one computation per (match_id, team)
    pairs = df[["match_id", "opponent", "date"]].drop_duplicates().copy()
    pairs["_pid"] = np.arange(len(pairs))

    opp_q = _opponent_quality(pairs, tms_q)

    pairs = pairs.merge(opp_q, on="_pid", how="left")
    df = df.merge(
        pairs[["match_id", "opponent", "opponent_points_l4",
               "opponent_cumulative_points", "opponent_form_flag"]],
        on=["match_id", "opponent"],
        how="left",
    )

    log.info("  opponent_points_l4 mean=%.2f  form_flag rate=%.1f%%",
             df["opponent_points_l4"].mean(), df["opponent_form_flag"].mean() * 100)
    _save_inter(df, "subs_with_opponent.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Outcomes
# ---------------------------------------------------------------------------

def build_step_7(df: pd.DataFrame, events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 7 — Outcomes")
    df = df.copy()
    goals = _build_goal_events(events, matches)

    gf15, ga15 = _goals_in_window(df, goals, "team", "minute", 15)
    df["goals_for_next15"]     = gf15
    df["goals_against_next15"] = ga15
    df["goal_diff_next15"]     = gf15 - ga15

    gf30, ga30 = _goals_in_window(df, goals, "team", "minute", 30)
    df["goals_for_next30"]     = gf30
    df["goals_against_next30"] = ga30
    df["goal_diff_next30"]     = gf30 - ga30

    df["xg_data_available"] = False

    last_ev_min = events.groupby("match_id")["minute"].max()
    df = df.merge(last_ev_min.rename("_last_min"), on="match_id", how="left")
    df["outcome_window_truncated"] = (df["minute"] + 30 > df["_last_min"].fillna(90)).astype(int)
    df = df.drop(columns=["_last_min"])

    log.info("  goal_diff_next15 mean=%.4f  truncated=%.1f%%",
             df["goal_diff_next15"].mean(), df["outcome_window_truncated"].mean() * 100)
    _save_inter(df, "subs_with_outcomes.parquet")
    return df


# ---------------------------------------------------------------------------
# Step 8 — Control pool
# ---------------------------------------------------------------------------

def build_step_8(
    subs_orig: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    tms: pd.DataFrame,
) -> pd.DataFrame:
    log.info("STEP 8 — Control pool")
    goals = _build_goal_events(events, matches)

    # All (match_id, team, league, season) with a sub in the data
    mt = subs_orig[["match_id", "team", "league", "season"]].drop_duplicates()

    # Cross-join with control minute range [55, 85]
    min_range = pd.DataFrame({"minute": range(CONTROL_MIN_START, CONTROL_MIN_END + 1), "_key": 1})
    controls = mt.assign(_key=1).merge(min_range, on="_key").drop(columns=["_key"])

    # Remove minutes where a sub actually occurred
    sub_mins = subs_orig[
        subs_orig["minute"].between(CONTROL_MIN_START, CONTROL_MIN_END)
    ][["match_id", "team", "minute"]].copy()
    sub_mins["_is_sub"] = True

    controls = controls.merge(sub_mins, on=["match_id", "team", "minute"], how="left")
    controls = controls[controls["_is_sub"].isna()].drop(columns=["_is_sub"]).reset_index(drop=True)
    log.info("  control rows: %d  (match-team pairs: %d)", len(controls), len(mt))

    # Match info
    controls = controls.merge(
        matches[["match_id", "home_team", "away_team", "date"]], on="match_id", how="left"
    )
    controls["date"] = pd.to_datetime(controls["date"])
    controls["is_home"] = (controls["team"] == controls["home_team"]).astype(int)
    controls["opponent"] = np.where(
        controls["team"] == controls["home_team"],
        controls["away_team"], controls["home_team"]
    )

    # Score at control minute
    gf_at, ga_at = _score_at_minute(controls, goals, "team", "minute")
    controls["goals_for_at_sub"]     = gf_at
    controls["goals_against_at_sub"] = ga_at
    controls["score_diff_at_sub"]    = gf_at - ga_at
    controls["is_winning"] = (controls["score_diff_at_sub"] > 0).astype(int)
    controls["is_level"]   = (controls["score_diff_at_sub"] == 0).astype(int)
    controls["is_losing"]  = (controls["score_diff_at_sub"] < 0).astype(int)
    controls["score_parse_error"] = 0
    controls["score_at_sub"] = np.nan

    # Sub slot at control minute (how many subs this team has already made before this minute)
    controls["_rid"] = np.arange(len(controls))
    ctrl_sub_join = controls[["_rid", "match_id", "team", "minute"]].merge(
        subs_orig[["match_id", "team", "minute"]].rename(columns={"minute": "sub_min"}),
        on=["match_id", "team"],
        how="left",
    )
    ctrl_sub_join["_before"] = ctrl_sub_join["sub_min"] < ctrl_sub_join["minute"]
    slot_agg = ctrl_sub_join.groupby("_rid")["_before"].sum()
    controls["sub_slot"] = (slot_agg.reindex(controls["_rid"], fill_value=0).values + 1).astype(int)
    controls = controls.drop(columns=["_rid"])

    # Player-specific columns are unknown for controls
    for col in ["player_in", "player_out", "player_in_position", "player_out_position",
                "player_in_quality_position", "player_in_quality_flag",
                "goals_per90", "assists_per90", "goal_contributions_per90",
                "player_out_minutes_l30", "player_out_minutes_l30_flag",
                "is_yellow_card_sub", "is_red_card_match"]:
        controls[col] = np.nan

    controls["sub_type"]      = np.nan
    controls["sub_direction"] = np.nan

    # Opponent quality
    tms_q = tms.copy()
    tms_q["points"] = tms_q["result"].map({"W": 3, "D": 1, "L": 0})
    pairs = controls[["match_id", "opponent", "date"]].drop_duplicates().copy()
    pairs["_pid"] = np.arange(len(pairs))
    opp_q = _opponent_quality(pairs, tms_q)
    pairs = pairs.merge(opp_q, on="_pid", how="left")
    controls = controls.merge(
        pairs[["match_id", "opponent", "opponent_points_l4",
               "opponent_cumulative_points", "opponent_form_flag"]],
        on=["match_id", "opponent"],
        how="left",
    )

    # Outcomes
    gf15, ga15 = _goals_in_window(controls, goals, "team", "minute", 15)
    controls["goals_for_next15"]     = gf15
    controls["goals_against_next15"] = ga15
    controls["goal_diff_next15"]     = gf15 - ga15

    gf30, ga30 = _goals_in_window(controls, goals, "team", "minute", 30)
    controls["goals_for_next30"]     = gf30
    controls["goals_against_next30"] = ga30
    controls["goal_diff_next30"]     = gf30 - ga30

    controls["xg_data_available"] = False
    last_ev_min = events.groupby("match_id")["minute"].max()
    controls = controls.merge(last_ev_min.rename("_last_min"), on="match_id", how="left")
    controls["outcome_window_truncated"] = (
        controls["minute"] + 30 > controls["_last_min"].fillna(90)
    ).astype(int)
    controls = controls.drop(columns=["_last_min"])

    controls["treated"] = 0
    _save_inter(controls, "control_pool.parquet")
    return controls


# ---------------------------------------------------------------------------
# Step 9 — Assemble model_ready.parquet
# ---------------------------------------------------------------------------

def build_step_9(treated: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 9 — Assemble model_ready")
    treated = treated.copy()
    treated["treated"] = 1

    # Align columns across both frames
    all_cols = sorted(set(treated.columns) | set(controls.columns))
    for col in all_cols:
        if col not in treated.columns:
            treated[col] = np.nan
        if col not in controls.columns:
            controls[col] = np.nan

    combined = pd.concat([treated, controls], ignore_index=True)

    # Ordered column layout
    id_cols      = ["match_id", "league", "season", "team", "minute", "treated"]
    score_cols   = ["score_at_sub", "goals_for_at_sub", "goals_against_at_sub",
                    "score_diff_at_sub", "is_winning", "is_level", "is_losing",
                    "score_parse_error"]
    match_cols   = ["is_home", "home_team", "away_team", "date", "opponent"]
    sub_cols     = ["player_in", "player_out", "sub_slot", "sub_type",
                    "is_yellow_card_sub", "is_red_card_match", "sub_direction",
                    "player_in_position", "player_out_position"]
    fatigue_cols = ["player_out_minutes_l30", "player_out_minutes_l30_flag"]
    quality_cols = ["player_in_quality_position", "goals_per90", "assists_per90",
                    "goal_contributions_per90", "player_in_quality_flag"]
    opp_cols     = ["opponent_points_l4", "opponent_cumulative_points", "opponent_form_flag"]
    outcome_cols = ["goals_for_next15", "goals_against_next15", "goal_diff_next15",
                    "goals_for_next30", "goals_against_next30", "goal_diff_next30",
                    "xg_data_available", "outcome_window_truncated"]

    ordered = [c for c in (id_cols + score_cols + match_cols + sub_cols +
                            fatigue_cols + quality_cols + opp_cols + outcome_cols)
               if c in combined.columns]
    remaining = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + remaining]

    path = PROCESSED_DIR / "model_ready.parquet"
    combined.to_parquet(path, index=False)

    n_treated  = (combined["treated"] == 1).sum()
    n_controls = (combined["treated"] == 0).sum()
    log.info("  treated=%d  control=%d  total=%d  cols=%d",
             n_treated, n_controls, len(combined), len(combined.columns))
    log.info("  null rates (key covariates):\n%s",
             combined[["score_diff_at_sub", "goals_per90",
                        "opponent_points_l4", "player_out_minutes_l30"]
                      ].isna().mean().round(4).mul(100).rename("% null").to_string())
    log.info("  saved model_ready.parquet")
    return combined


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    log.info("========== Phase 2 — Feature Engineering START ==========")

    subs, events, matches, pm, pss, tms = _load_raw()

    df = build_step_1(subs, matches)
    df = build_step_2(df, events)
    df = build_step_3(df, pm, pss)
    df = build_step_4(df, pm, matches)
    df = build_step_5(df, pss)
    df = build_step_6(df, matches, tms)
    df = build_step_7(df, events, matches)

    controls = build_step_8(subs, events, matches, tms)

    result = build_step_9(df, controls)

    log.info("========== Phase 2 — Feature Engineering COMPLETE ==========")
    return result


if __name__ == "__main__":
    run_all()
