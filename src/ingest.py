"""
Phase 1 — Data Ingestion
Collects all raw data needed for causal analysis of substitution timing.
All outputs saved to data/raw/. No joins, filters, or transformations
except the Table 7 crosswalk.
"""

import os

# Must be set before any soccerdata import
os.environ["SOCCERDATA_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"
)

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# 2 most recently completed full seasons
SEASONS = ["2022-23", "2023-24"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_validate(df: pd.DataFrame, path: Path, key_fields: list[str]) -> None:
    """Save dataframe to parquet and print validation summary."""
    df.to_parquet(path, index=False)
    log.info("Saved %s  (%d rows)", path.name, len(df))
    print(f"\n{'='*60}")
    print(f"FILE    : {path.name}")
    print(f"ROWS    : {len(df):,}")
    print(f"COLUMNS : {list(df.columns)}")
    print("SAMPLE (3 rows):")
    print(df.head(3).to_string())
    print("NULL COUNTS (key join fields):")
    for field in key_fields:
        if field in df.columns:
            null_count = df[field].isna().sum()
            null_pct = null_count / max(len(df), 1) * 100
            flag = "  *** WARNING: >1% nulls ***" if null_pct > 1.0 else ""
            print(f"  {field}: {null_count} nulls ({null_pct:.2f}%){flag}")
        else:
            print(f"  {field}: COLUMN NOT PRESENT")
    print(f"{'='*60}\n")


def _get_fbref() -> object:
    """Return a soccerdata FBref client."""
    import soccerdata as sd
    return sd.FBref(leagues=LEAGUES, seasons=SEASONS)


def _parse_minute(m) -> int | None:
    """Parse FBref minute strings like '23’ or '90+3’ into integers.
    FBref uses U+2019 RIGHT SINGLE QUOTATION MARK, not ASCII apostrophe.
    """
    if pd.isna(m):
        return None
    if isinstance(m, (int, float)):
        return int(m)
    # Strip both ASCII apostrophe and Unicode right single quote (U+2019)
    s = str(m).replace("’", "").replace("'", "").strip()
    if not s:
        return None
    if "+" in s:
        parts = s.split("+")
        try:
            return int(parts[0]) + int(parts[1])
        except ValueError:
            try:
                return int(parts[0])
            except ValueError:
                return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Table collectors
# ---------------------------------------------------------------------------

def collect_table_1() -> pd.DataFrame | None:
    """
    Collect match metadata from FBref.

    Confounders enabled: C3 (home/away status).
    Saves to: data/raw/fbref_matches.parquet
    Grain: one row per match.

    Actual soccerdata columns: league, season, game, date, home_team, score,
    away_team, game_id, etc.
    match_id uses 'game' string (e.g. '2022-08-05 Crystal Palace-Arsenal')
    to be consistent with all other tables.
    """
    log.info("TABLE 1 — Match metadata")
    try:
        fbref = _get_fbref()
        schedule = fbref.read_schedule().reset_index()

        df = schedule.rename(columns={"game": "match_id", "game_id": "fbref_id"}).copy()

        # Parse score column (format: '0–3' with en-dash) into home/away scores
        if "score" in df.columns:
            parsed = df["score"].str.split("–", expand=True)
            if parsed.shape[1] >= 2:
                df["home_score"] = pd.to_numeric(parsed[0], errors="coerce")
                df["away_score"] = pd.to_numeric(parsed[1], errors="coerce")

        keep = [f for f in ["match_id", "fbref_id", "date", "home_team", "away_team",
                             "home_score", "away_score", "league", "season"]
                if f in df.columns]
        df = df[keep].dropna(subset=["match_id"]).copy()

        path = RAW_DIR / "fbref_matches.parquet"
        _save_and_validate(df, path, key_fields=["match_id"])
        return df

    except Exception as exc:
        log.error("TABLE 1 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_2() -> pd.DataFrame | None:
    """
    Collect substitution events from FBref.

    Treatment variable: minute (exact minute of substitution).
    Confounders enabled: C6 via player_out, C5 via player_in, C7 via sub_slot,
    C1 via score_at_sub (actual scoreline at moment of substitution).
    Saves to: data/raw/fbref_substitutions.parquet
    Grain: one row per substitution event.

    Actual soccerdata columns: league, season, game, team, minute, score,
    player1 (coming in), player2 (coming off), event_type.
    """
    log.info("TABLE 2 — Substitution events")
    try:
        fbref = _get_fbref()
        events = fbref.read_events().reset_index()

        # Filter to substitute_in events only
        subs = events[events["event_type"] == "substitute_in"].copy()
        log.info("TABLE 2: %d substitute_in events found", len(subs))

        subs = subs.rename(columns={
            "game":    "match_id",
            "player1": "player_in",
            "player2": "player_out",
            "score":   "score_at_sub",
        })

        subs["minute"] = subs["minute"].apply(_parse_minute)

        subs = subs.sort_values(["match_id", "team", "minute"])
        subs["sub_slot"] = subs.groupby(["match_id", "team"]).cumcount() + 1

        keep = [f for f in ["match_id", "league", "season", "team", "minute",
                             "player_in", "player_out", "score_at_sub", "sub_slot"]
                if f in subs.columns]
        df = subs[keep].copy()

        path = RAW_DIR / "fbref_substitutions.parquet"
        _save_and_validate(df, path, key_fields=["match_id", "minute"])
        return df

    except Exception as exc:
        log.error("TABLE 2 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_3() -> pd.DataFrame | None:
    """
    Collect match event timeline (goals, own goals, cards) from FBref.

    Confounders enabled: C1 (scoreline reconstruction via score column).
    Also enables disciplinary sub classification and red-card match flagging (Phase 2).
    Saves to: data/raw/fbref_match_events.parquet
    Grain: one row per event.

    Actual soccerdata columns: league, season, game, team, minute, score,
    player1 (primary player), player2 (secondary), event_type.
    """
    log.info("TABLE 3 — Match events timeline")
    try:
        fbref = _get_fbref()
        events = fbref.read_events().reset_index()

        df = events.rename(columns={
            "game":    "match_id",
            "player1": "player",
            "player2": "player2",
        }).copy()

        df["minute"] = df["minute"].apply(_parse_minute)
        df["event_type"] = df["event_type"].str.lower().str.replace(" ", "_", regex=False)

        # Keep goals and cards only — exclude substitutes
        target_types = {"goal", "own_goal", "yellow_card", "red_card",
                        "yellow_red_card", "red_card_second_yellow"}
        df = df[df["event_type"].isin(target_types)].copy()

        keep = [f for f in ["match_id", "league", "season", "minute",
                             "event_type", "team", "player", "score"]
                if f in df.columns]
        df = df[keep].copy()

        path = RAW_DIR / "fbref_match_events.parquet"
        _save_and_validate(df, path, key_fields=["match_id", "minute"])
        return df

    except Exception as exc:
        log.error("TABLE 3 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_4() -> pd.DataFrame | None:
    """
    Collect per-player per-match minutes played from FBref.

    Confounders enabled: C6 (player fatigue — rolling 30-day minutes, computed Phase 2).
    Saves to: data/raw/fbref_player_minutes.parquet
    Grain: one row per player per match appearance.

    Actual soccerdata columns: league, season, game, jersey_number, player,
    team, is_starter, position, minutes_played.
    """
    log.info("TABLE 4 — Player minutes log")
    try:
        fbref = _get_fbref()
        lineups = fbref.read_lineup().reset_index()

        df = lineups.rename(columns={"game": "match_id"}).copy()

        keep = [f for f in ["match_id", "league", "season", "player", "team",
                             "minutes_played", "position", "is_starter"]
                if f in df.columns]
        df = df[keep].copy()

        path = RAW_DIR / "fbref_player_minutes.parquet"
        _save_and_validate(df, path, key_fields=["match_id", "player"])
        return df

    except Exception as exc:
        log.error("TABLE 4 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_5() -> pd.DataFrame | None:
    """
    Collect player season statistics from FBref.

    Confounders enabled: C5 (substitute player quality — xG per 90,
    goals per 90, position for sub-type classification).
    Saves to: data/raw/fbref_player_season_stats.parquet
    Grain: one row per player per season.

    Valid stat_types in this soccerdata version:
    ['standard', 'keeper', 'shooting', 'playing_time', 'misc']
    """
    log.info("TABLE 5 — Player season statistics")
    try:
        fbref = _get_fbref()

        std = fbref.read_player_season_stats(stat_type="standard").reset_index()
        log.info("TABLE 5: standard stats columns: %s", list(std.columns))

        # Flatten multi-level columns — actual names after flattening:
        # 'Playing Time_Min', 'Performance_Gls', 'Performance_Ast', 'pos', etc.
        if isinstance(std.columns[0], tuple):
            std.columns = ["_".join(filter(None, c)).strip("_") for c in std.columns]

        df = std.rename(columns={
            "pos":               "position",
            "Playing Time_Min":  "minutes_played_season",
            "Performance_Gls":   "goals_season",
            "Performance_Ast":   "assists_season",
        })

        mins = pd.to_numeric(df.get("minutes_played_season"), errors="coerce")
        if mins is not None and mins.max() < 100:
            df["minutes_played_season"] = mins * 90

        # Try to get xG from shooting stat type
        try:
            shooting = fbref.read_player_season_stats(stat_type="shooting").reset_index()
            if isinstance(shooting.columns[0], tuple):
                shooting.columns = ["_".join(filter(None, c)).strip("_") for c in shooting.columns]
            xg_col = next((c for c in shooting.columns if "xg" in c.lower()), None)
            if xg_col:
                join_cols = [c for c in ["player", "team", "season"] if c in shooting.columns]
                shooting = shooting[join_cols + [xg_col]].rename(columns={xg_col: "xg_season"})
                df = df.merge(shooting, on=join_cols, how="left")
                log.info("TABLE 5: xG column '%s' merged from shooting stats", xg_col)
            else:
                log.warning("TABLE 5: no xG column found in shooting stats")
        except Exception as exc:
            log.warning("TABLE 5: could not merge shooting stats: %s", exc)

        if "minutes_played_season" in df.columns:
            mins = pd.to_numeric(df["minutes_played_season"], errors="coerce")
            df["player_quality_flag"] = (mins < 270).astype(int)

        keep = [f for f in ["player", "team", "season", "league", "position",
                             "minutes_played_season", "goals_season", "assists_season",
                             "xg_season", "player_quality_flag"]
                if f in df.columns]
        df = df[keep].drop_duplicates().copy()

        path = RAW_DIR / "fbref_player_season_stats.parquet"
        _save_and_validate(df, path, key_fields=["player"])
        return df

    except Exception as exc:
        log.error("TABLE 5 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_6() -> pd.DataFrame | None:
    """
    Collect match-level team season stats (goals, wins, draws) from FBref.

    Replaces original Understat shot-level xG plan — Understat changed their
    site architecture in 2025 making the understat library permanently broken.
    Game state at sub-time will be reconstructed from actual scoreline using
    Tables 1 and 3 (goals with exact minutes) in Phase 2.

    This table provides match-level team context (venue, result, opponent)
    useful for descriptive analysis.
    Saves to: data/raw/fbref_team_match_stats.parquet
    Grain: one row per team per match.
    """
    log.info("TABLE 6 — Team match stats (FBref)")
    try:
        fbref = _get_fbref()
        df = fbref.read_team_match_stats(stat_type="schedule").reset_index()

        cols_lower = {c.lower(): c for c in df.columns}

        def _find(candidates):
            for c in candidates:
                if c in cols_lower:
                    return cols_lower[c]
            return None

        rename = {}
        for target, candidates in [
            ("match_id",  ["game_id", "game", "match_id"]),
            ("date",      ["date"]),
            ("team",      ["team", "squad"]),
            ("opponent",  ["opponent", "opp"]),
            ("venue",     ["venue"]),
            ("result",    ["result"]),
            ("goals_for", ["gf"]),
            ("goals_against", ["ga"]),
            ("league",    ["league", "competition"]),
            ("season",    ["season"]),
        ]:
            src = _find(candidates)
            if src and src != target:
                rename[src] = target

        df = df.rename(columns=rename)

        keep = [f for f in ["match_id", "date", "team", "opponent", "venue",
                             "result", "goals_for", "goals_against", "league", "season"]
                if f in df.columns]
        df = df[keep].dropna(subset=["date"]).copy()

        for col in ["goals_for", "goals_against"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        path = RAW_DIR / "fbref_team_match_stats.parquet"
        _save_and_validate(df, path, key_fields=["match_id", "team"])
        return df

    except Exception as exc:
        log.error("TABLE 6 FAILED: %s", exc, exc_info=True)
        return None


def collect_table_7() -> None:
    """
    Crosswalk table — no longer needed.

    Originally built FBref-to-Understat match ID links. Removed because
    Understat changed their site architecture in 2025, making their data
    inaccessible. Game state confounder (C2) will use actual scoreline
    reconstructed from Tables 1 and 3 in Phase 2.
    """
    log.info("TABLE 7 — Crosswalk skipped (Understat unavailable)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all() -> None:
    """Run all collection functions in sequence."""
    log.info("========== Phase 1 — Data Ingestion START ==========")
    log.info("Leagues : %s", LEAGUES)
    log.info("Seasons : %s", SEASONS)

    collect_table_1()
    collect_table_2()
    collect_table_3()
    collect_table_4()
    collect_table_5()
    collect_table_6()
    collect_table_7()

    log.info("========== Phase 1 — Data Ingestion COMPLETE ==========")


if __name__ == "__main__":
    run_all()
