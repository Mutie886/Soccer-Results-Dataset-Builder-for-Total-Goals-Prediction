
import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soccer Results Dataset Builder", layout="wide")

DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
MASTER_PATH = DATA_DIR / "matches_master.csv"
FEATURES_PATH = DATA_DIR / "matches_features.csv"
STANDINGS_PATH = DATA_DIR / "standings_history.csv"
REJECTED_PATH = DATA_DIR / "rejected_duplicates.csv"
STATE_PATH = DATA_DIR / "system_state.json"

MASTER_COLUMNS = [
    "match_id",
    "cycle_id",
    "week_number",
    "batch_id",
    "batch_match_number",
    "global_order",
    "home_team",
    "home_goals",
    "away_goals",
    "away_team",
    "total_goals",
    "result",
    "goal_diff",
    "match_key",
    "created_at",
]

FEATURE_ORDER_COLUMNS = ["global_order"]
PRIOR_WEEK_ORDER_COLUMNS = ["cycle_id", "week_number", "global_order"]

EXPECTED_HISTORY_COLUMNS = [
    "goals_for",
    "goals_against",
    "team_result",
    "venue",
    "total_goals",
    "cycle_id",
    "week_number",
    "batch_match_number",
    "global_order",
    "match_id",
]

STANDINGS_COLUMNS = [
    "cycle_id",
    "week_number",
    "team",
    "rank",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_diff",
    "points",
    "form_last5",
    "form_points_last5",
    "form_wins_last5",
    "form_draws_last5",
    "form_losses_last5",
]


# ----------------------- helpers -----------------------
def normalize_team_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).title()


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def total_goal_class(total_goals: int) -> str:
    return str(total_goals) if total_goals <= 6 else "6_plus"


def stable_batch_hash(raw_text: str, week_number: int, batch_id: str) -> str:
    payload = f"{week_number}|{batch_id}|{raw_text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_system() -> None:
    for path in [MASTER_PATH, FEATURES_PATH, STANDINGS_PATH, REJECTED_PATH, STATE_PATH]:
        if path.exists():
            path.unlink()
    for key in ["last_processed_hash", "last_processed_result"]:
        if key in st.session_state:
            del st.session_state[key]


def read_master() -> pd.DataFrame:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH)
        for col in MASTER_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[MASTER_COLUMNS]
        for col in [
            "match_id", "cycle_id", "week_number", "batch_match_number", "global_order",
            "home_goals", "away_goals", "total_goals", "goal_diff"
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame(columns=MASTER_COLUMNS)


def save_master(df: pd.DataFrame) -> None:
    df.to_csv(MASTER_PATH, index=False)


def save_features(df: pd.DataFrame) -> None:
    df.to_csv(FEATURES_PATH, index=False)


def save_standings(df: pd.DataFrame) -> None:
    df.to_csv(STANDINGS_PATH, index=False)


def append_rejected(df: pd.DataFrame) -> None:
    if df.empty:
        return
    header = not REJECTED_PATH.exists()
    df.to_csv(REJECTED_PATH, mode="a", header=header, index=False)


def parse_week_number_from_text(raw_text: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", raw_text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def is_noise_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", low):
        return True
    if low.startswith("english league"):
        return True
    if "week" in low and "#" in low:
        return True
    if low.startswith("league "):
        return True
    return False


def week_header_value(line: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", str(line), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def split_input_into_week_sections(raw_text: str, fallback_week_number: int) -> List[Tuple[int, List[str]]]:
    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines()]
    sections: List[Tuple[int, List[str]]] = []
    current_week: Optional[int] = None
    current_lines: List[str] = []

    for line in raw_lines:
        if not line:
            continue
        detected_week = week_header_value(line)
        if detected_week is not None and line.lower().startswith("english league"):
            if current_lines:
                sections.append((int(current_week if current_week is not None else fallback_week_number), current_lines))
                current_lines = []
            current_week = detected_week
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", line.lower()):
            continue
        if is_noise_line(line):
            continue
        if current_week is None:
            current_week = int(fallback_week_number)
        current_lines.append(line)

    if current_lines:
        sections.append((int(current_week if current_week is not None else fallback_week_number), current_lines))

    return sections


def parse_matches(raw_text: str, week_number: int, batch_id: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    sections = split_input_into_week_sections(raw_text, int(week_number))

    if not sections:
        return pd.DataFrame(), ["No usable match lines were found after cleaning."]

    records = []
    if len(sections) > 1:
        warnings.append(f"Detected {len(sections)} week sections in this input and assigned week numbers per section.")

    chronological_sections = list(reversed(sections))  # input is pasted newest-to-oldest

    running_block_counter = 0
    for section_week, section_lines in chronological_sections:
        remainder = len(section_lines) % 4
        if remainder:
            warnings.append(
                f"Week {section_week}: ignored the last {remainder} line(s) because a valid match needs 4 lines."
            )
            section_lines = section_lines[: len(section_lines) - remainder]

        section_records = []
        for i in range(0, len(section_lines), 4):
            running_block_counter += 1
            home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = section_lines[i:i+4]
            home_team = normalize_team_name(home_team_raw)
            away_team = normalize_team_name(away_team_raw)

            if home_team == away_team:
                warnings.append(f"Week {section_week}, block {running_block_counter}: home team and away team are identical.")
                continue

            try:
                home_goals = int(home_goals_raw)
                away_goals = int(away_goals_raw)
            except ValueError:
                warnings.append(f"Week {section_week}, block {running_block_counter}: scores must be integers.")
                continue

            if home_goals < 0 or away_goals < 0:
                warnings.append(f"Week {section_week}, block {running_block_counter}: negative goals are not allowed.")
                continue

            section_records.append(
                {
                    "batch_id": batch_id,
                    "week_number": int(section_week),
                    "home_team": home_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "away_team": away_team,
                    "total_goals": home_goals + away_goals,
                    "result": result_code(home_goals, away_goals),
                    "goal_diff": home_goals - away_goals,
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
            )

        if not section_records:
            continue

        section_df = pd.DataFrame(section_records)
        section_df = section_df.iloc[::-1].reset_index(drop=True)  # top pasted match is latest -> store it last
        records.extend(section_df.to_dict("records"))

    df = pd.DataFrame(records)
    if df.empty:
        return df, warnings

    before = len(df)
    df["_batch_dedupe_key"] = (
        df["week_number"].astype(str) + "|" +
        df["home_team"] + "|" + df["away_team"] + "|" +
        df["home_goals"].astype(str) + "|" + df["away_goals"].astype(str)
    )
    df = df.drop_duplicates(subset=["_batch_dedupe_key"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        warnings.append(f"Removed {removed} duplicate match(es) inside this pasted input.")

    df["batch_match_number"] = np.arange(1, len(df) + 1)
    return df.drop(columns=["_batch_dedupe_key"]), warnings


def assign_cycle_ids(master: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if new_df.empty:
        return new_df.copy()

    out = new_df.copy().reset_index(drop=True)
    if master.empty:
        current_cycle = 1
        prev_week = None
    else:
        master_sorted = master.copy()
        master_sorted["global_order"] = pd.to_numeric(master_sorted["global_order"], errors="coerce")
        master_sorted = master_sorted.sort_values("global_order")
        cycle_vals = pd.to_numeric(master_sorted["cycle_id"], errors="coerce").dropna()
        week_vals = pd.to_numeric(master_sorted["week_number"], errors="coerce").dropna()
        current_cycle = int(cycle_vals.iloc[-1]) if not cycle_vals.empty else 1
        prev_week = int(week_vals.iloc[-1]) if not week_vals.empty else None

    assigned_cycles = []
    for _, row in out.iterrows():
        current_week = int(row["week_number"])
        if prev_week is not None and current_week < prev_week:
            current_cycle += 1
        assigned_cycles.append(current_cycle)
        prev_week = current_week

    out["cycle_id"] = assigned_cycles
    return out


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def rate_from_series(series: pd.Series, value: str) -> float:
    return float((series == value).mean()) if len(series) else np.nan


def over_rate(series: pd.Series, threshold: float) -> float:
    return float((series > threshold).mean()) if len(series) else np.nan


def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_HISTORY_COLUMNS)


def compute_team_history(prior_matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if prior_matches.empty:
        return empty_history_df()

    home_part = prior_matches[prior_matches["home_team"] == team].copy()
    if not home_part.empty:
        home_part["goals_for"] = home_part["home_goals"]
        home_part["goals_against"] = home_part["away_goals"]
        home_part["team_result"] = np.where(
            home_part["home_goals"] > home_part["away_goals"], "W",
            np.where(home_part["home_goals"] < home_part["away_goals"], "L", "D")
        )
        home_part["venue"] = "home"

    away_part = prior_matches[prior_matches["away_team"] == team].copy()
    if not away_part.empty:
        away_part["goals_for"] = away_part["away_goals"]
        away_part["goals_against"] = away_part["home_goals"]
        away_part["team_result"] = np.where(
            away_part["away_goals"] > away_part["home_goals"], "W",
            np.where(away_part["away_goals"] < away_part["home_goals"], "L", "D")
        )
        away_part["venue"] = "away"

    hist = pd.concat([home_part, away_part], ignore_index=True)
    if hist.empty:
        return empty_history_df()

    for col in EXPECTED_HISTORY_COLUMNS:
        if col not in hist.columns:
            hist[col] = np.nan
    hist = hist.sort_values(PRIOR_WEEK_ORDER_COLUMNS).reset_index(drop=True)
    return hist[EXPECTED_HISTORY_COLUMNS]


def summary_features_from_history(history: pd.DataFrame, prefix: str) -> dict:
    feats = {}

    def add_window(name: str, h: pd.DataFrame) -> None:
        feats[f"{prefix}_{name}_avg_scored"] = safe_mean(h["goals_for"])
        feats[f"{prefix}_{name}_avg_conceded"] = safe_mean(h["goals_against"])
        feats[f"{prefix}_{name}_avg_total_goals"] = safe_mean(h["total_goals"])
        feats[f"{prefix}_{name}_win_rate"] = rate_from_series(h["team_result"], "W")
        feats[f"{prefix}_{name}_draw_rate"] = rate_from_series(h["team_result"], "D")
        feats[f"{prefix}_{name}_loss_rate"] = rate_from_series(h["team_result"], "L")
        feats[f"{prefix}_{name}_over_1_5_rate"] = over_rate(h["total_goals"], 1.5)
        feats[f"{prefix}_{name}_over_2_5_rate"] = over_rate(h["total_goals"], 2.5)
        feats[f"{prefix}_{name}_over_3_5_rate"] = over_rate(h["total_goals"], 3.5)

    add_window("last3", history.tail(3))
    add_window("last5", history.tail(5))
    add_window("last10", history.tail(10))

    home_only = history[history["venue"] == "home"].tail(5)
    away_only = history[history["venue"] == "away"].tail(5)
    feats[f"{prefix}_home_last5_avg_scored"] = safe_mean(home_only["goals_for"])
    feats[f"{prefix}_home_last5_avg_conceded"] = safe_mean(home_only["goals_against"])
    feats[f"{prefix}_away_last5_avg_scored"] = safe_mean(away_only["goals_for"])
    feats[f"{prefix}_away_last5_avg_conceded"] = safe_mean(away_only["goals_against"])
    feats[f"{prefix}_matches_played"] = int(len(history))
    return feats


def head_to_head_features(prior_matches: pd.DataFrame, home_team: str, away_team: str) -> dict:
    if prior_matches.empty:
        return {
            "h2h_last3_avg_total_goals": np.nan,
            "h2h_home_team_win_rate": np.nan,
            "h2h_draw_rate": np.nan,
            "h2h_away_team_win_rate": np.nan,
            "h2h_matches_played": 0,
        }

    h2h = prior_matches[
        ((prior_matches["home_team"] == home_team) & (prior_matches["away_team"] == away_team)) |
        ((prior_matches["home_team"] == away_team) & (prior_matches["away_team"] == home_team))
    ].copy()
    h2h = h2h.sort_values(PRIOR_WEEK_ORDER_COLUMNS).tail(3)

    if h2h.empty:
        return {
            "h2h_last3_avg_total_goals": np.nan,
            "h2h_home_team_win_rate": np.nan,
            "h2h_draw_rate": np.nan,
            "h2h_away_team_win_rate": np.nan,
            "h2h_matches_played": 0,
        }

    outcomes = []
    for _, row in h2h.iterrows():
        if row["home_goals"] == row["away_goals"]:
            outcomes.append("D")
        elif row["home_team"] == home_team:
            outcomes.append("H" if row["home_goals"] > row["away_goals"] else "A")
        else:
            outcomes.append("H" if row["away_goals"] > row["home_goals"] else "A")

    out = pd.Series(outcomes)
    return {
        "h2h_last3_avg_total_goals": safe_mean(h2h["total_goals"]),
        "h2h_home_team_win_rate": rate_from_series(out, "H"),
        "h2h_draw_rate": rate_from_series(out, "D"),
        "h2h_away_team_win_rate": rate_from_series(out, "A"),
        "h2h_matches_played": int(len(h2h)),
    }


def compute_standings_history(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    df = master_df.copy()
    for col in ["cycle_id", "week_number", "global_order", "home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["global_order"]).reset_index(drop=True)

    standings_rows = []
    stats_by_cycle = {}
    ordered_weeks = df[["cycle_id", "week_number"]].drop_duplicates().sort_values(["cycle_id", "week_number"])

    for _, wk in ordered_weeks.iterrows():
        cycle_id = int(wk["cycle_id"])
        week_number = int(wk["week_number"])

        if cycle_id not in stats_by_cycle:
            stats_by_cycle[cycle_id] = {}

        cycle_stats = stats_by_cycle[cycle_id]
        week_matches = df[(df["cycle_id"] == cycle_id) & (df["week_number"] == week_number)].copy()

        for _, match in week_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            hg = int(match["home_goals"])
            ag = int(match["away_goals"])

            for team in [home_team, away_team]:
                if team not in cycle_stats:
                    cycle_stats[team] = {
                        "played": 0, "wins": 0, "draws": 0, "losses": 0,
                        "goals_for": 0, "goals_against": 0, "points": 0,
                        "form": [],
                    }

            # home
            cycle_stats[home_team]["played"] += 1
            cycle_stats[home_team]["goals_for"] += hg
            cycle_stats[home_team]["goals_against"] += ag

            # away
            cycle_stats[away_team]["played"] += 1
            cycle_stats[away_team]["goals_for"] += ag
            cycle_stats[away_team]["goals_against"] += hg

            if hg > ag:
                cycle_stats[home_team]["wins"] += 1
                cycle_stats[home_team]["points"] += 3
                cycle_stats[home_team]["form"].append("W")

                cycle_stats[away_team]["losses"] += 1
                cycle_stats[away_team]["form"].append("L")
            elif hg < ag:
                cycle_stats[away_team]["wins"] += 1
                cycle_stats[away_team]["points"] += 3
                cycle_stats[away_team]["form"].append("W")

                cycle_stats[home_team]["losses"] += 1
                cycle_stats[home_team]["form"].append("L")
            else:
                cycle_stats[home_team]["draws"] += 1
                cycle_stats[away_team]["draws"] += 1
                cycle_stats[home_team]["points"] += 1
                cycle_stats[away_team]["points"] += 1
                cycle_stats[home_team]["form"].append("D")
                cycle_stats[away_team]["form"].append("D")

        table = []
        for team, stt in cycle_stats.items():
            form_last5_list = stt["form"][-5:]
            form_string = "".join(form_last5_list)
            form_points = sum(3 if x == "W" else 1 if x == "D" else 0 for x in form_last5_list)
            table.append(
                {
                    "cycle_id": cycle_id,
                    "week_number": week_number,
                    "team": team,
                    "played": int(stt["played"]),
                    "wins": int(stt["wins"]),
                    "draws": int(stt["draws"]),
                    "losses": int(stt["losses"]),
                    "goals_for": int(stt["goals_for"]),
                    "goals_against": int(stt["goals_against"]),
                    "goal_diff": int(stt["goals_for"] - stt["goals_against"]),
                    "points": int(stt["points"]),
                    "form_last5": form_string,
                    "form_points_last5": int(form_points),
                    "form_wins_last5": int(sum(x == "W" for x in form_last5_list)),
                    "form_draws_last5": int(sum(x == "D" for x in form_last5_list)),
                    "form_losses_last5": int(sum(x == "L" for x in form_last5_list)),
                }
            )

        week_table = pd.DataFrame(table)
        if week_table.empty:
            continue
        week_table = week_table.sort_values(
            ["points", "goal_diff", "goals_for", "team"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        week_table.insert(3, "rank", np.arange(1, len(week_table) + 1))
        standings_rows.append(week_table[STANDINGS_COLUMNS])

    if not standings_rows:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    standings = pd.concat(standings_rows, ignore_index=True)
    return standings


def get_latest_prior_standing(standings_df: pd.DataFrame, team: str, cycle_id: int, week_number: int) -> Optional[pd.Series]:
    if standings_df.empty:
        return None

    prior = standings_df[
        (standings_df["team"] == team) &
        (
            (standings_df["cycle_id"] < cycle_id) |
            ((standings_df["cycle_id"] == cycle_id) & (standings_df["week_number"] < week_number))
        )
    ].copy()

    if prior.empty:
        return None

    prior = prior.sort_values(["cycle_id", "week_number"])
    return prior.iloc[-1]


def standings_feature_dict(standing: Optional[pd.Series], prefix: str) -> dict:
    if standing is None:
        return {
            f"{prefix}_rank": np.nan,
            f"{prefix}_points": np.nan,
            f"{prefix}_goal_diff": np.nan,
            f"{prefix}_played": np.nan,
            f"{prefix}_form_points_last5": np.nan,
            f"{prefix}_form_wins_last5": np.nan,
            f"{prefix}_form_draws_last5": np.nan,
            f"{prefix}_form_losses_last5": np.nan,
        }

    return {
        f"{prefix}_rank": float(standing["rank"]),
        f"{prefix}_points": float(standing["points"]),
        f"{prefix}_goal_diff": float(standing["goal_diff"]),
        f"{prefix}_played": float(standing["played"]),
        f"{prefix}_form_points_last5": float(standing["form_points_last5"]),
        f"{prefix}_form_wins_last5": float(standing["form_wins_last5"]),
        f"{prefix}_form_draws_last5": float(standing["form_draws_last5"]),
        f"{prefix}_form_losses_last5": float(standing["form_losses_last5"]),
    }


def build_feature_dataset(master_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()
    for col in ["cycle_id", "week_number", "batch_match_number", "global_order", "match_id", "home_goals", "away_goals", "total_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["global_order"]).reset_index(drop=True)

    if standings_df is None or standings_df.empty:
        standings_df = pd.DataFrame(columns=STANDINGS_COLUMNS)
    else:
        standings_df = standings_df.copy()
        standings_df["cycle_id"] = pd.to_numeric(standings_df["cycle_id"], errors="coerce")
        standings_df["week_number"] = pd.to_numeric(standings_df["week_number"], errors="coerce")

    rows = []
    for _, row in df.iterrows():
        prior = df[
            (df["cycle_id"] < row["cycle_id"]) |
            ((df["cycle_id"] == row["cycle_id"]) & (df["week_number"] < row["week_number"]))
        ].copy()

        home_hist = compute_team_history(prior, row["home_team"])
        away_hist = compute_team_history(prior, row["away_team"])
        home_standing = get_latest_prior_standing(standings_df, row["home_team"], int(row["cycle_id"]), int(row["week_number"]))
        away_standing = get_latest_prior_standing(standings_df, row["away_team"], int(row["cycle_id"]), int(row["week_number"]))

        feats = {
            "match_id": int(row["match_id"]),
            "cycle_id": int(row["cycle_id"]),
            "week_number": int(row["week_number"]),
            "batch_match_number": int(row["batch_match_number"]),
            "global_order": int(row["global_order"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
        feats.update(summary_features_from_history(home_hist, "home_team"))
        feats.update(summary_features_from_history(away_hist, "away_team"))
        feats.update(head_to_head_features(prior, row["home_team"], row["away_team"]))
        feats.update(standings_feature_dict(home_standing, "home_team"))
        feats.update(standings_feature_dict(away_standing, "away_team"))

        feats["standings_rank_gap"] = (
            feats.get("home_team_rank", np.nan) - feats.get("away_team_rank", np.nan)
            if pd.notna(feats.get("home_team_rank", np.nan)) and pd.notna(feats.get("away_team_rank", np.nan))
            else np.nan
        )
        feats["standings_points_gap"] = (
            feats.get("home_team_points", np.nan) - feats.get("away_team_points", np.nan)
            if pd.notna(feats.get("home_team_points", np.nan)) and pd.notna(feats.get("away_team_points", np.nan))
            else np.nan
        )
        feats["standings_goal_diff_gap"] = (
            feats.get("home_team_goal_diff", np.nan) - feats.get("away_team_goal_diff", np.nan)
            if pd.notna(feats.get("home_team_goal_diff", np.nan)) and pd.notna(feats.get("away_team_goal_diff", np.nan))
            else np.nan
        )
        feats["standings_form_points_gap"] = (
            feats.get("home_team_form_points_last5", np.nan) - feats.get("away_team_form_points_last5", np.nan)
            if pd.notna(feats.get("home_team_form_points_last5", np.nan)) and pd.notna(feats.get("away_team_form_points_last5", np.nan))
            else np.nan
        )

        feats["target_total_goals"] = int(row["total_goals"])
        feats["target_total_class"] = total_goal_class(int(row["total_goals"]))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["match_id", "cycle_id", "week_number", "batch_match_number", "global_order", "target_total_goals"]
    feature_num_cols = [c for c in numeric_cols if c not in exclude]
    for col in feature_num_cols:
        miss = feat_df[col].isna().astype(int)
        if miss.any():
            feat_df[f"{col}_missing"] = miss
            feat_df[col] = feat_df[col].fillna(0.0)

    return feat_df


def append_to_master(new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame, pd.DataFrame]:
    master = read_master()
    new_df = assign_cycle_ids(master, new_df)

    new_df["match_key"] = (
        new_df["cycle_id"].astype(int).astype(str) + "|" +
        new_df["week_number"].astype(int).astype(str) + "|" +
        new_df["home_team"] + "|" + new_df["away_team"] + "|" +
        new_df["home_goals"].astype(int).astype(str) + "|" + new_df["away_goals"].astype(int).astype(str)
    )

    existing_keys = set(master["match_key"].astype(str)) if not master.empty else set()
    duplicate_mask = new_df["match_key"].astype(str).isin(existing_keys)
    rejected_existing = new_df[duplicate_mask].copy()
    accepted = new_df[~duplicate_mask].copy()

    if accepted.empty:
        standings = compute_standings_history(master)
        features = build_feature_dataset(master, standings)
        if not standings.empty:
            save_standings(standings)
        if not features.empty:
            save_features(features)
        return master, rejected_existing, 0, features, standings

    next_match_id = 1 if master.empty else int(pd.to_numeric(master["match_id"], errors="coerce").dropna().max()) + 1
    next_global_order = 1 if master.empty else int(pd.to_numeric(master["global_order"], errors="coerce").dropna().max()) + 1

    accepted = accepted.reset_index(drop=True)
    accepted.insert(0, "match_id", range(next_match_id, next_match_id + len(accepted)))
    accepted.insert(5, "global_order", range(next_global_order, next_global_order + len(accepted)))

    for col in MASTER_COLUMNS:
        if col not in accepted.columns:
            accepted[col] = np.nan

    master = pd.concat([master, accepted[MASTER_COLUMNS]], ignore_index=True)
    for c in ["match_id", "cycle_id", "week_number", "batch_match_number", "global_order"]:
        master[c] = pd.to_numeric(master[c], errors="coerce")
    master = master.sort_values(FEATURE_ORDER_COLUMNS).reset_index(drop=True)
    save_master(master)

    if not rejected_existing.empty:
        append_rejected(rejected_existing)

    standings = compute_standings_history(master)
    save_standings(standings)

    features = build_feature_dataset(master, standings)
    if not features.empty:
        save_features(features)

    return master, rejected_existing, len(accepted), features, standings


def get_notification_metrics() -> dict:
    master = read_master()
    features = pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame()
    standings = pd.read_csv(STANDINGS_PATH) if STANDINGS_PATH.exists() else pd.DataFrame()
    return {
        "master_rows": int(len(master)),
        "feature_rows": int(len(features)),
        "teams_seen": int(len(pd.unique(pd.concat([master["home_team"], master["away_team"]], ignore_index=True)))) if not master.empty else 0,
        "cycles_seen": int(pd.to_numeric(master["cycle_id"], errors="coerce").dropna().max()) if not master.empty and pd.to_numeric(master["cycle_id"], errors="coerce").dropna().size else 0,
        "standings_rows": int(len(standings)),
    }


# ----------------------- UI -----------------------
st.title("⚽ Soccer Results Dataset Builder")
st.caption("Clean raw soccer results, detect week sections, compute standings automatically from results, and export model-ready datasets.")

left, right = st.columns([4, 1])
with left:
    raw_text = st.text_area("Paste raw input", height=320, placeholder="Paste the raw match results here...")
with right:
    detected_week = parse_week_number_from_text(raw_text) if raw_text else None
    week_number = st.number_input("Fallback week number", min_value=1, max_value=38, value=int(detected_week) if detected_week else 1, step=1)
    batch_id = st.text_input("Batch id", value="batch_manual")
    process = st.button("Process and save", type="primary", use_container_width=True)
    refresh = st.button("Refresh system / start new dataset", use_container_width=True)

if refresh:
    reset_system()
    st.success("System refreshed. All saved records were cleared.")

st.subheader("Notifications dashboard")
metrics = get_notification_metrics()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Master rows", metrics["master_rows"])
m2.metric("Feature rows", metrics["feature_rows"])
m3.metric("Teams seen", metrics["teams_seen"])
m4.metric("Cycles seen", metrics["cycles_seen"])
m5.metric("Standings rows", metrics["standings_rows"])

if process:
    if not raw_text.strip():
        st.error("Paste some raw input first.")
    else:
        batch_hash = stable_batch_hash(raw_text, int(week_number), batch_id.strip() or "batch_manual")
        if st.session_state.get("last_processed_hash") == batch_hash:
            last = st.session_state.get("last_processed_result", {})
            st.warning("This exact batch was already processed. No records were added again.")
            if last:
                st.info(
                    f"Last result: accepted {last.get('accepted', 0)}, existing duplicates {last.get('existing_duplicates', 0)}, warnings {last.get('warnings', 0)}."
                )
        else:
            parsed_df, warnings = parse_matches(raw_text, int(week_number), batch_id.strip() or "batch_manual")
            for msg in warnings:
                st.warning(msg)

            if parsed_df.empty:
                st.error("No valid matches were found after cleaning.")
            else:
                master_df, rejected_existing, accepted_count, features_df, standings_df = append_to_master(parsed_df)
                st.session_state["last_processed_hash"] = batch_hash
                st.session_state["last_processed_result"] = {
                    "accepted": int(accepted_count),
                    "existing_duplicates": int(len(rejected_existing)),
                    "warnings": int(len(warnings)),
                }

                if accepted_count > 0:
                    st.success(f"Saved {accepted_count} new match(es).")
                else:
                    st.info("All cleaned matches from this batch were already present in the saved history.")
                if len(rejected_existing) > 0:
                    st.info(f"Ignored {len(rejected_existing)} match(es) already present in saved history.")

master_download = MASTER_PATH.read_bytes() if MASTER_PATH.exists() else b""
features_download = FEATURES_PATH.read_bytes() if FEATURES_PATH.exists() else b""
standings_download = STANDINGS_PATH.read_bytes() if STANDINGS_PATH.exists() else b""

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        "Download matches_master.csv",
        data=master_download,
        file_name="matches_master.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not bool(master_download),
    )
with d2:
    st.download_button(
        "Download matches_features.csv",
        data=features_download,
        file_name="matches_features.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not bool(features_download),
    )
with d3:
    st.download_button(
        "Download standings_history.csv",
        data=standings_download,
        file_name="standings_history.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not bool(standings_download),
    )
