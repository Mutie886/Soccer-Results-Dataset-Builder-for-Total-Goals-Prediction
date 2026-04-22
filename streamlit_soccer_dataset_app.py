import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Soccer Total Goals Predictor", layout="wide")

# =========================
# Storage paths
# =========================
DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
MASTER_PATH = DATA_DIR / "matches_master.csv"
FEATURES_PATH = DATA_DIR / "matches_features.csv"
STANDINGS_PATH = DATA_DIR / "standings_history.csv"
REJECTED_PATH = DATA_DIR / "rejected_duplicates.csv"
STATE_PATH = DATA_DIR / "system_state.json"
MODEL_BUNDLE_PATH = DATA_DIR / "model_bundle.joblib"

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

RESULT_CLASS_MAP = {"H": "1", "D": "X", "A": "2"}
RESULT_CLASS_ORDER = ["1", "X", "2"]
TOTAL_CLASS_ORDER = ["0", "1", "2", "3", "4", "5", "6"]
MIN_ROWS_TO_TRAIN = 60


# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#0f172a 0%,#111827 100%); color: #e5e7eb;}
    .main-card {background: rgba(17,24,39,0.88); border: 1px solid rgba(148,163,184,0.18); border-radius: 18px; padding: 18px 18px 14px 18px; margin-bottom: 14px; box-shadow: 0 14px 30px rgba(0,0,0,0.24);}
    .section-title {font-size: 1.08rem; font-weight: 700; margin-bottom: 0.55rem; color: #f8fafc;}
    .caption-small {font-size: 0.83rem; color: #cbd5e1; margin-top: 4px;}
    .prediction-card {background: rgba(15,23,42,0.92); border: 1px solid rgba(96,165,250,0.20); border-radius: 18px; padding: 16px; min-height: 338px; margin-bottom: 14px;}
    .fixture-title {font-size: 1.05rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;}
    .subhead {font-size: 0.83rem; font-weight: 700; color: #93c5fd; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em;}
    .prob-grid {display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; margin-bottom: 8px;}
    .prob-grid.total {grid-template-columns: repeat(4, minmax(0,1fr));}
    .prob-chip {background: rgba(30,41,59,0.95); border: 1px solid rgba(148,163,184,0.18); border-radius: 12px; padding: 8px 10px; text-align: center;}
    .prob-label {font-size: 0.8rem; color: #cbd5e1;}
    .prob-value {font-size: 0.95rem; font-weight: 800; color: #f8fafc; margin-top: 2px;}
    .pick-line {margin-top: 8px; font-size: 0.86rem; color: #e2e8f0;}
    .metric-note {font-size: 0.82rem; color: #cbd5e1; margin-top: 2px;}
    .hr-soft {height: 1px; border: 0; background: rgba(148,163,184,0.14); margin: 8px 0 10px 0;}
    div[data-testid="stMetric"] {background: rgba(17,24,39,0.88); border: 1px solid rgba(148,163,184,0.18); padding: 14px 14px 10px 14px; border-radius: 16px;}
    .small-help {font-size:0.80rem; color:#cbd5e1; line-height:1.35;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_team_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).title()


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def total_goal_class(total_goals: int) -> str:
    return str(total_goals) if total_goals <= 6 else "6"


def parse_week_number_from_text(raw_text: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", raw_text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def week_header_value(line: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", str(line), flags=re.IGNORECASE)
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


def stable_hash(*parts: str) -> str:
    payload = "|".join(part.strip() for part in parts)
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
    for path in [MASTER_PATH, FEATURES_PATH, STANDINGS_PATH, REJECTED_PATH, STATE_PATH, MODEL_BUNDLE_PATH]:
        if path.exists():
            path.unlink()
    for key in [
        "last_results_hash",
        "last_results_result",
        "last_predict_hash",
    ]:
        st.session_state.pop(key, None)


# =========================
# IO
# =========================
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


def read_features() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return pd.DataFrame()


def read_standings() -> pd.DataFrame:
    if STANDINGS_PATH.exists():
        return pd.read_csv(STANDINGS_PATH)
    return pd.DataFrame(columns=STANDINGS_COLUMNS)


def load_model_bundle() -> Optional[dict]:
    if MODEL_BUNDLE_PATH.exists():
        try:
            return joblib.load(MODEL_BUNDLE_PATH)
        except Exception:
            return None
    return None


def save_model_bundle(bundle: dict) -> None:
    # Save only pickle-safe objects. The trained sklearn pipelines and plain metadata
    # are serializable once all helper callables are top-level functions.
    safe_bundle = {
        "result_rf_model": bundle.get("result_rf_model"),
        "result_lr_model": bundle.get("result_lr_model"),
        "total_rf_model": bundle.get("total_rf_model"),
        "total_lr_model": bundle.get("total_lr_model"),
        "feature_columns": list(bundle.get("feature_columns", [])),
        "metrics": dict(bundle.get("metrics", {})),
    }
    joblib.dump(safe_bundle, MODEL_BUNDLE_PATH)


# =========================
# Parsing recent results input
# =========================
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

    records: List[dict] = []
    if len(sections) > 1:
        warnings.append(f"Detected {len(sections)} week sections in this input and assigned week numbers per section.")

    # Input is pasted newest-to-oldest. Process bottom sections first for chronology.
    chronological_sections = list(reversed(sections))
    running_block_counter = 0

    for section_week, section_lines in chronological_sections:
        remainder = len(section_lines) % 4
        if remainder:
            warnings.append(f"Week {section_week}: ignored the last {remainder} line(s) because a valid match needs 4 lines.")
            section_lines = section_lines[: len(section_lines) - remainder]

        section_records: List[dict] = []
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
                    "created_at": now_iso(),
                }
            )

        if not section_records:
            continue

        section_df = pd.DataFrame(section_records)
        # Top match is latest, so within each week store it last.
        section_df = section_df.iloc[::-1].reset_index(drop=True)
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


# =========================
# Feature building
# =========================
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
    hist = hist.sort_values(["cycle_id", "week_number", "global_order"]).reset_index(drop=True)
    return hist[EXPECTED_HISTORY_COLUMNS]


def summary_features_from_history(history: pd.DataFrame, prefix: str) -> dict:
    feats: Dict[str, float] = {}

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
    default = {
        "h2h_last3_avg_total_goals": np.nan,
        "h2h_home_team_win_rate": np.nan,
        "h2h_draw_rate": np.nan,
        "h2h_away_team_win_rate": np.nan,
        "h2h_matches_played": 0,
    }
    if prior_matches.empty:
        return default

    h2h = prior_matches[
        ((prior_matches["home_team"] == home_team) & (prior_matches["away_team"] == away_team)) |
        ((prior_matches["home_team"] == away_team) & (prior_matches["away_team"] == home_team))
    ].copy()
    h2h = h2h.sort_values(["cycle_id", "week_number", "global_order"]).tail(3)
    if h2h.empty:
        return default

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
    stats_by_cycle: Dict[int, dict] = {}
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
                        "goals_for": 0, "goals_against": 0, "points": 0, "form": []
                    }

            cycle_stats[home_team]["played"] += 1
            cycle_stats[home_team]["goals_for"] += hg
            cycle_stats[home_team]["goals_against"] += ag
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
        week_table = week_table.sort_values(["points", "goal_diff", "goals_for", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
        week_table.insert(3, "rank", np.arange(1, len(week_table) + 1))
        standings_rows.append(week_table[STANDINGS_COLUMNS])

    if not standings_rows:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)
    return pd.concat(standings_rows, ignore_index=True)


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


def get_latest_snapshot_standing(standings_df: pd.DataFrame, team: str) -> Optional[pd.Series]:
    if standings_df.empty:
        return None
    one = standings_df[standings_df["team"] == team].copy()
    if one.empty:
        return None
    one = one.sort_values(["cycle_id", "week_number"])
    return one.iloc[-1]


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


def add_gap_features(feats: dict) -> dict:
    feats["standings_rank_gap"] = feats.get("home_team_rank", np.nan) - feats.get("away_team_rank", np.nan) if pd.notna(feats.get("home_team_rank", np.nan)) and pd.notna(feats.get("away_team_rank", np.nan)) else np.nan
    feats["standings_points_gap"] = feats.get("home_team_points", np.nan) - feats.get("away_team_points", np.nan) if pd.notna(feats.get("home_team_points", np.nan)) and pd.notna(feats.get("away_team_points", np.nan)) else np.nan
    feats["standings_goal_diff_gap"] = feats.get("home_team_goal_diff", np.nan) - feats.get("away_team_goal_diff", np.nan) if pd.notna(feats.get("home_team_goal_diff", np.nan)) and pd.notna(feats.get("away_team_goal_diff", np.nan)) else np.nan
    feats["standings_form_points_gap"] = feats.get("home_team_form_points_last5", np.nan) - feats.get("away_team_form_points_last5", np.nan) if pd.notna(feats.get("home_team_form_points_last5", np.nan)) and pd.notna(feats.get("away_team_form_points_last5", np.nan)) else np.nan
    return feats


def fill_missing_and_flags(feat_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["match_id", "cycle_id", "week_number", "batch_match_number", "global_order", "target_total_goals"]
    feature_num_cols = [c for c in numeric_cols if c not in exclude]
    for col in feature_num_cols:
        miss = feat_df[col].isna().astype(int)
        if miss.any():
            feat_df[f"{col}_missing"] = miss
            feat_df[col] = feat_df[col].fillna(0.0)
    return feat_df


def build_feature_dataset(master_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()
    for col in ["cycle_id", "week_number", "batch_match_number", "global_order", "match_id", "home_goals", "away_goals", "total_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["global_order"]).reset_index(drop=True)

    standings_df = standings_df.copy() if standings_df is not None else pd.DataFrame(columns=STANDINGS_COLUMNS)
    if not standings_df.empty:
        standings_df["cycle_id"] = pd.to_numeric(standings_df["cycle_id"], errors="coerce")
        standings_df["week_number"] = pd.to_numeric(standings_df["week_number"], errors="coerce")

    rows = []
    for _, row in df.iterrows():
        prior = df[(df["cycle_id"] < row["cycle_id"]) | ((df["cycle_id"] == row["cycle_id"]) & (df["week_number"] < row["week_number"]))].copy()
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
        add_gap_features(feats)
        feats["target_total_goals"] = int(row["total_goals"])
        feats["target_total_class"] = total_goal_class(int(row["total_goals"]))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    return fill_missing_and_flags(feat_df)


def build_prediction_feature_row(master_df: pd.DataFrame, standings_df: pd.DataFrame, home_team: str, away_team: str, next_cycle: int, next_week: int) -> pd.DataFrame:
    prior = master_df.copy()
    if not prior.empty:
        for col in ["cycle_id", "week_number", "batch_match_number", "global_order", "match_id", "home_goals", "away_goals", "total_goals"]:
            prior[col] = pd.to_numeric(prior[col], errors="coerce")
        prior = prior.sort_values(["global_order"]).reset_index(drop=True)

    home_hist = compute_team_history(prior, home_team)
    away_hist = compute_team_history(prior, away_team)
    home_standing = get_latest_snapshot_standing(standings_df, home_team)
    away_standing = get_latest_snapshot_standing(standings_df, away_team)

    feats = {
        "match_id": -1,
        "cycle_id": int(next_cycle),
        "week_number": int(next_week),
        "batch_match_number": 0,
        "global_order": (int(pd.to_numeric(master_df["global_order"], errors="coerce").dropna().max()) + 1) if not master_df.empty and pd.to_numeric(master_df["global_order"], errors="coerce").dropna().size else 1,
        "home_team": home_team,
        "away_team": away_team,
    }
    feats.update(summary_features_from_history(home_hist, "home_team"))
    feats.update(summary_features_from_history(away_hist, "away_team"))
    feats.update(head_to_head_features(prior, home_team, away_team))
    feats.update(standings_feature_dict(home_standing, "home_team"))
    feats.update(standings_feature_dict(away_standing, "away_team"))
    add_gap_features(feats)
    row = pd.DataFrame([feats])
    row = fill_missing_and_flags(row)
    return row


# =========================
# Append results and rebuild datasets
# =========================
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
        save_standings(standings)
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
    master = master.sort_values(["global_order"]).reset_index(drop=True)
    save_master(master)

    if not rejected_existing.empty:
        append_rejected(rejected_existing)

    standings = compute_standings_history(master)
    save_standings(standings)
    features = build_feature_dataset(master, standings)
    save_features(features)
    return master, rejected_existing, len(accepted), features, standings


# =========================
# Model training
# =========================
def latest_cycle_week(master_df: pd.DataFrame) -> Tuple[int, int]:
    if master_df.empty:
        return 1, 1
    sorted_df = master_df.sort_values("global_order")
    row = sorted_df.iloc[-1]
    return int(row["cycle_id"]), int(row["week_number"])


def next_cycle_week(master_df: pd.DataFrame) -> Tuple[int, int]:
    cycle_id, week_number = latest_cycle_week(master_df)
    if week_number >= 38:
        return cycle_id + 1, 1
    return cycle_id, week_number + 1


def prepare_training_matrix(features_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = features_df.copy().sort_values("global_order").reset_index(drop=True)
    result_map_df = master_df[["match_id", "result"]].copy()
    merged = df.merge(result_map_df, on="match_id", how="left")
    merged["result_target"] = merged["result"].map(RESULT_CLASS_MAP)
    merged["total_target"] = merged["target_total_class"].replace({"6_plus": "6"}).astype(str)

    drop_cols = {
        "match_id", "cycle_id", "week_number", "batch_match_number", "global_order",
        "target_total_goals", "target_total_class", "result", "result_target", "total_target"
    }
    feature_cols = [c for c in merged.columns if c not in drop_cols]
    X = merged[feature_cols].copy()
    y_result = merged["result_target"].astype(str)
    y_total = merged["total_target"].astype(str)
    return X, y_result, y_total




# Pickle-safe column selectors for sklearn ColumnTransformer
def numeric_feature_selector(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_feature_selector(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def build_rf_classifier() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=0.0), numeric_feature_selector),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_feature_selector),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=320,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def build_lr_classifier() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric_feature_selector),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_feature_selector),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        max_iter=1400,
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def weighted_average_probabilities(probs_a: np.ndarray, probs_b: np.ndarray, weight_a: float, weight_b: float) -> np.ndarray:
    total = max(weight_a + weight_b, 1e-9)
    out = (weight_a * probs_a + weight_b * probs_b) / total
    denom = out.sum()
    return out / denom if denom > 0 else np.repeat(1.0 / len(out), len(out))


def holdout_weight_from_accuracy(acc: float, floor: float = 0.15, ceiling: float = 0.85) -> float:
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        return 0.50
    return float(np.clip(acc, floor, ceiling))

def accuracy_to_weight(acc: float) -> float:
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        return 0.50
    return float(np.clip(0.32 + 0.55 * acc, 0.35, 0.72))


def train_models(features_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[Optional[dict], List[str]]:
    warnings: List[str] = []
    if features_df.empty or master_df.empty:
        warnings.append("No data available for model training yet.")
        return None, warnings
    if len(features_df) < MIN_ROWS_TO_TRAIN:
        warnings.append(f"Need at least {MIN_ROWS_TO_TRAIN} feature rows before training. Current rows: {len(features_df)}.")
        return None, warnings

    X, y_result, y_total = prepare_training_matrix(features_df, master_df)
    if y_result.nunique() < 3:
        warnings.append("Result model still needs all three classes (1, X, 2) represented.")
        return None, warnings
    if y_total.nunique() < 4:
        warnings.append("Total-goals model needs more class variety before training.")
        return None, warnings

    ordered = features_df.sort_values("global_order").reset_index(drop=True)
    split_idx = max(int(len(ordered) * 0.80), len(ordered) - max(20, len(ordered) // 5))
    split_idx = min(max(split_idx, 40), len(ordered) - 10)
    train_idx = ordered.index[:split_idx]
    test_idx = ordered.index[split_idx:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    yr_train = y_result.iloc[train_idx]
    yr_test = y_result.iloc[test_idx]
    yt_train = y_total.iloc[train_idx]
    yt_test = y_total.iloc[test_idx]

    result_rf_eval = build_rf_classifier()
    result_lr_eval = build_lr_classifier()
    total_rf_eval = build_rf_classifier()
    total_lr_eval = build_lr_classifier()

    result_rf_eval.fit(X_train, yr_train)
    result_lr_eval.fit(X_train, yr_train)
    total_rf_eval.fit(X_train, yt_train)
    total_lr_eval.fit(X_train, yt_train)

    result_rf_acc = accuracy_score(yr_test, result_rf_eval.predict(X_test)) if len(X_test) else np.nan
    result_lr_acc = accuracy_score(yr_test, result_lr_eval.predict(X_test)) if len(X_test) else np.nan
    total_rf_acc = accuracy_score(yt_test, total_rf_eval.predict(X_test)) if len(X_test) else np.nan
    total_lr_acc = accuracy_score(yt_test, total_lr_eval.predict(X_test)) if len(X_test) else np.nan

    # Final fit on full data
    result_rf_model = build_rf_classifier()
    result_lr_model = build_lr_classifier()
    total_rf_model = build_rf_classifier()
    total_lr_model = build_lr_classifier()
    result_rf_model.fit(X, y_result)
    result_lr_model.fit(X, y_result)
    total_rf_model.fit(X, y_total)
    total_lr_model.fit(X, y_total)

    bundle = {
        "result_rf_model": result_rf_model,
        "result_lr_model": result_lr_model,
        "total_rf_model": total_rf_model,
        "total_lr_model": total_lr_model,
        "feature_columns": list(X.columns),
        "metrics": {
            "result_rf_accuracy": None if np.isnan(result_rf_acc) else float(result_rf_acc),
            "result_lr_accuracy": None if np.isnan(result_lr_acc) else float(result_lr_acc),
            "total_rf_accuracy": None if np.isnan(total_rf_acc) else float(total_rf_acc),
            "total_lr_accuracy": None if np.isnan(total_lr_acc) else float(total_lr_acc),
            "result_accuracy": None if np.isnan(np.nanmean([result_rf_acc, result_lr_acc])) else float(np.nanmean([result_rf_acc, result_lr_acc])),
            "total_accuracy": None if np.isnan(np.nanmean([total_rf_acc, total_lr_acc])) else float(np.nanmean([total_rf_acc, total_lr_acc])),
            "result_rf_weight": holdout_weight_from_accuracy(result_rf_acc),
            "result_lr_weight": holdout_weight_from_accuracy(result_lr_acc),
            "total_rf_weight": holdout_weight_from_accuracy(total_rf_acc),
            "total_lr_weight": holdout_weight_from_accuracy(total_lr_acc),
            "result_model_weight": accuracy_to_weight(float(np.nanmean([result_rf_acc, result_lr_acc]))),
            "total_model_weight": accuracy_to_weight(float(np.nanmean([total_rf_acc, total_lr_acc]))),
            "training_rows": int(len(X)),
            "trained_at": now_iso(),
        },
    }
    save_model_bundle(bundle)
    return bundle, warnings


# =========================
# Prediction input parsing
# =========================
def parse_prediction_input(raw_text: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines() if str(ln).strip()]
    if not lines:
        return pd.DataFrame(), ["No prediction lines were found."]

    expected_labels = ["1", "X", "2", "0", "1", "2", "3", "4", "5", "6"]
    block_size = 22
    remainder = len(lines) % block_size
    if remainder:
        warnings.append(f"Ignored the last {remainder} line(s) because each prediction block needs 22 non-empty lines.")
        lines = lines[: len(lines) - remainder]

    records = []
    for i in range(0, len(lines), block_size):
        block = lines[i:i+block_size]
        if len(block) < block_size:
            continue
        home_team = normalize_team_name(block[0])
        away_team = normalize_team_name(block[1])
        labels = [str(x).strip().upper() if str(x).strip().upper() == "X" else str(x).strip() for x in block[2::2]]
        odds_vals = block[3::2]

        if labels != expected_labels:
            warnings.append(f"Skipped prediction block {(i // block_size) + 1}: market labels do not match the expected order.")
            continue
        try:
            odds = [float(x) for x in odds_vals]
        except ValueError:
            warnings.append(f"Skipped prediction block {(i // block_size) + 1}: odds must be numeric.")
            continue

        if any(o <= 0 for o in odds):
            warnings.append(f"Skipped prediction block {(i // block_size) + 1}: odds must be positive.")
            continue

        records.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "odd_1": odds[0],
                "odd_X": odds[1],
                "odd_2": odds[2],
                "odd_total_0": odds[3],
                "odd_total_1": odds[4],
                "odd_total_2": odds[5],
                "odd_total_3": odds[6],
                "odd_total_4": odds[7],
                "odd_total_5": odds[8],
                "odd_total_6": odds[9],
            }
        )

    return pd.DataFrame(records), warnings


def normalized_inverse_odds(odds: List[float]) -> np.ndarray:
    arr = np.array([1.0 / max(o, 1e-9) for o in odds], dtype=float)
    denom = arr.sum()
    if denom <= 0:
        return np.repeat(1.0 / len(arr), len(arr))
    return arr / denom


def align_prediction_row(row: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = row.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0 if col.endswith("_missing") else np.nan
    aligned = aligned[feature_columns]
    for col in aligned.columns:
        if aligned[col].dtype == object:
            aligned[col] = aligned[col].fillna("Unknown")
    return aligned


def model_probs_to_order(classes: np.ndarray, probs: np.ndarray, desired_order: List[str]) -> np.ndarray:
    mapping = {str(c): float(p) for c, p in zip(classes, probs)}
    return np.array([mapping.get(label, 0.0) for label in desired_order], dtype=float)


def blend_probabilities(model_probs: np.ndarray, market_probs: Optional[np.ndarray], model_weight: float) -> np.ndarray:
    if market_probs is None:
        out = model_probs.copy()
    else:
        out = model_weight * model_probs + (1.0 - model_weight) * market_probs
    denom = out.sum()
    return out / denom if denom > 0 else np.repeat(1.0 / len(out), len(out))


def generate_predictions(pred_input_df: pd.DataFrame, master_df: pd.DataFrame, standings_df: pd.DataFrame, bundle: dict) -> List[dict]:
    predictions = []
    next_cycle, next_week = next_cycle_week(master_df)
    for _, rec in pred_input_df.iterrows():
        home_team = rec["home_team"]
        away_team = rec["away_team"]
        row = build_prediction_feature_row(master_df, standings_df, home_team, away_team, next_cycle, next_week)
        aligned = align_prediction_row(row, bundle["feature_columns"])

        result_rf = bundle["result_rf_model"]
        result_lr = bundle["result_lr_model"]
        total_rf = bundle["total_rf_model"]
        total_lr = bundle["total_lr_model"]

        result_rf_probs = model_probs_to_order(result_rf.classes_, result_rf.predict_proba(aligned)[0], RESULT_CLASS_ORDER)
        result_lr_probs = model_probs_to_order(result_lr.classes_, result_lr.predict_proba(aligned)[0], RESULT_CLASS_ORDER)
        total_rf_probs = model_probs_to_order(total_rf.classes_, total_rf.predict_proba(aligned)[0], TOTAL_CLASS_ORDER)
        total_lr_probs = model_probs_to_order(total_lr.classes_, total_lr.predict_proba(aligned)[0], TOTAL_CLASS_ORDER)

        result_model_probs = weighted_average_probabilities(
            result_rf_probs, result_lr_probs,
            bundle["metrics"].get("result_rf_weight", 0.5),
            bundle["metrics"].get("result_lr_weight", 0.5),
        )
        total_model_probs = weighted_average_probabilities(
            total_rf_probs, total_lr_probs,
            bundle["metrics"].get("total_rf_weight", 0.5),
            bundle["metrics"].get("total_lr_weight", 0.5),
        )

        market_result_probs = normalized_inverse_odds([rec["odd_1"], rec["odd_X"], rec["odd_2"]])
        market_total_probs = normalized_inverse_odds([
            rec["odd_total_0"], rec["odd_total_1"], rec["odd_total_2"], rec["odd_total_3"],
            rec["odd_total_4"], rec["odd_total_5"], rec["odd_total_6"],
        ])

        # Use market more strongly when the trained model is still modest on holdout data.
        result_probs = blend_probabilities(result_model_probs, market_result_probs, bundle["metrics"].get("result_model_weight", 0.5))
        total_probs = blend_probabilities(total_model_probs, market_total_probs, bundle["metrics"].get("total_model_weight", 0.5))

        predictions.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "result_probs": {k: float(v) for k, v in zip(RESULT_CLASS_ORDER, result_probs)},
                "total_probs": {k: float(v) for k, v in zip(TOTAL_CLASS_ORDER, total_probs)},
                "best_result": RESULT_CLASS_ORDER[int(np.argmax(result_probs))],
                "best_total": TOTAL_CLASS_ORDER[int(np.argmax(total_probs))],
            }
        )
    return predictions


# =========================
# Notifications and rendering
# =========================
def get_notification_metrics() -> dict:
    master = read_master()
    features = read_features()
    standings = read_standings()
    bundle = load_model_bundle()
    latest_cycle, latest_week = latest_cycle_week(master) if not master.empty else (0, 0)
    return {
        "master_rows": int(len(master)),
        "feature_rows": int(len(features)),
        "standings_rows": int(len(standings)),
        "teams_seen": int(len(pd.unique(pd.concat([master["home_team"], master["away_team"]], ignore_index=True)))) if not master.empty else 0,
        "cycles_seen": int(pd.to_numeric(master["cycle_id"], errors="coerce").dropna().max()) if not master.empty and pd.to_numeric(master["cycle_id"], errors="coerce").dropna().size else 0,
        "latest_cycle": latest_cycle,
        "latest_week": latest_week,
        "result_acc": None if not bundle else bundle["metrics"].get("result_accuracy"),
        "total_acc": None if not bundle else bundle["metrics"].get("total_accuracy"),
        "trained_at": None if not bundle else bundle["metrics"].get("trained_at"),
        "training_rows": 0 if not bundle else bundle["metrics"].get("training_rows", 0),
    }


def fmt_pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


def render_prediction_cards(predictions: List[dict]) -> None:
    if not predictions:
        return
    st.markdown('<div class="main-card"><div class="section-title">Prediction dashboard</div><div class="caption-small">Percentages combine the trained model with the current market signal when odds are provided.</div></div>', unsafe_allow_html=True)
    per_row = 2
    for i in range(0, len(predictions), per_row):
        cols = st.columns(per_row)
        for col, pred in zip(cols, predictions[i:i+per_row]):
            with col:
                result_probs = pred["result_probs"]
                total_probs = pred["total_probs"]
                result_html = "".join(
                    f'<div class="prob-chip"><div class="prob-label">{label}</div><div class="prob-value">{fmt_pct(result_probs[label])}</div></div>'
                    for label in RESULT_CLASS_ORDER
                )
                total_html = "".join(
                    f'<div class="prob-chip"><div class="prob-label">{label}</div><div class="prob-value">{fmt_pct(total_probs[label])}</div></div>'
                    for label in TOTAL_CLASS_ORDER
                )
                st.markdown(
                    f'''
                    <div class="prediction-card">
                        <div class="fixture-title">{html.escape(pred['home_team'])} vs {html.escape(pred['away_team'])}</div>
                        <div class="subhead">Match result probabilities</div>
                        <div class="prob-grid">{result_html}</div>
                        <div class="subhead">Total goals probabilities</div>
                        <div class="prob-grid total">{total_html}</div>
                        <div class="pick-line"><strong>Most likely result:</strong> {pred['best_result']}</div>
                        <div class="pick-line"><strong>Most likely total goals:</strong> {pred['best_total']}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )


# =========================
# UI
# =========================
st.title("⚽ Soccer Total Goals Prediction System")
st.caption("Train from continuous results updates, retrain after each week update, and make optional on-demand multiclass predictions.")

# Controls area
left, right = st.columns([1.15, 1.0], gap="large")
with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1) Recent matches input</div><div class="small-help">Paste recent match results here. The system cleans the text, stores it in chronological order, rebuilds standings and features, then retrains the models from the updated system data.</div>', unsafe_allow_html=True)
    results_raw_text = st.text_area(
        "Recent results input",
        height=320,
        placeholder="Paste the recent results here...",
        key="results_text",
        label_visibility="collapsed",
    )
    st.markdown('<hr class="hr-soft"/>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        detected_week = parse_week_number_from_text(results_raw_text) if results_raw_text else None
        fallback_week_number = st.number_input("Fallback week number", min_value=1, max_value=38, value=int(detected_week) if detected_week else 1, step=1)
    with c2:
        batch_id = st.text_input("Batch id", value="batch_manual")
    r1, r2 = st.columns([1, 1])
    with r1:
        process_results = st.button("Process results & retrain", type="primary", use_container_width=True)
    with r2:
        refresh_system = st.button("Refresh system / start new dataset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2) Optional prediction input</div><div class="small-help">Paste fixture pairs with their 1 / X / 2 odds and exact total-goals odds. This does not interfere with results ingestion or model retraining. Use it only when you want predictions.</div>', unsafe_allow_html=True)
    prediction_raw_text = st.text_area(
        "Prediction input",
        height=320,
        placeholder="Paste the prediction input here...",
        key="prediction_text",
        label_visibility="collapsed",
    )
    p1, p2 = st.columns([1, 1])
    with p1:
        run_prediction = st.button("Predict now", use_container_width=True)
    with p2:
        retrain_only = st.button("Retrain from saved data", use_container_width=True)
    st.markdown('<div class="caption-small">Prediction uses the latest trained models. If odds are pasted, they are blended with the model probabilities to produce practical percentages.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if refresh_system:
    reset_system()
    st.success("System refreshed. Saved records, standings, features, and trained models were cleared.")

# Notifications dashboard
st.markdown('<div class="main-card"><div class="section-title">Notifications dashboard</div></div>', unsafe_allow_html=True)
metrics = get_notification_metrics()
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Master rows", metrics["master_rows"])
mc2.metric("Feature rows", metrics["feature_rows"])
mc3.metric("Standings rows", metrics["standings_rows"])
mc4.metric("Teams seen", metrics["teams_seen"])
mc5.metric("Cycles seen", metrics["cycles_seen"])
mc6, mc7, mc8, mc9 = st.columns(4)
mc6.metric("Latest cycle", metrics["latest_cycle"])
mc7.metric("Latest week", metrics["latest_week"])
mc8.metric("Result accuracy", "-" if metrics["result_acc"] is None else f"{100*metrics['result_acc']:.1f}%")
mc9.metric("Total-goals accuracy", "-" if metrics["total_acc"] is None else f"{100*metrics['total_acc']:.1f}%")
if metrics["trained_at"]:
    st.caption(f"Latest trained model: {metrics['trained_at']} | training rows: {metrics['training_rows']}")
else:
    st.caption("No trained model saved yet.")

# Process results and retrain
if process_results:
    if not results_raw_text.strip():
        st.error("Paste some recent results first.")
    else:
        current_hash = stable_hash(results_raw_text, str(fallback_week_number), batch_id or "batch_manual")
        if st.session_state.get("last_results_hash") == current_hash:
            last = st.session_state.get("last_results_result", {})
            st.warning("This exact results batch was already processed. No records were added again.")
            if last:
                st.info(f"Last result: accepted {last.get('accepted', 0)}, existing duplicates {last.get('existing_duplicates', 0)}, warnings {last.get('warnings', 0)}.")
        else:
            parsed_df, parse_warnings = parse_matches(results_raw_text, int(fallback_week_number), batch_id.strip() or "batch_manual")
            for msg in parse_warnings:
                st.warning(msg)
            if parsed_df.empty:
                st.error("No valid matches were found after cleaning.")
            else:
                master_df, rejected_existing, accepted_count, features_df, standings_df = append_to_master(parsed_df)
                bundle = None
                train_warnings: List[str] = []
                if accepted_count > 0:
                    bundle, train_warnings = train_models(features_df, master_df)
                    for msg in train_warnings:
                        st.warning(msg)

                st.session_state["last_results_hash"] = current_hash
                st.session_state["last_results_result"] = {
                    "accepted": int(accepted_count),
                    "existing_duplicates": int(len(rejected_existing)),
                    "warnings": int(len(parse_warnings) + len(train_warnings)),
                }
                if accepted_count > 0:
                    st.success(f"Saved {accepted_count} new match(es), rebuilt datasets, and refreshed training.")
                else:
                    st.info("All cleaned matches from this batch were already present in the saved history.")
                if len(rejected_existing) > 0:
                    st.info(f"Ignored {len(rejected_existing)} match(es) already present in saved history.")
                if bundle is not None:
                    st.success("Models retrained successfully.")

# Manual retrain button
if retrain_only:
    master_df = read_master()
    features_df = read_features()
    if master_df.empty or features_df.empty:
        st.error("No saved training data found yet.")
    else:
        bundle, train_warnings = train_models(features_df, master_df)
        for msg in train_warnings:
            st.warning(msg)
        if bundle is not None:
            st.success("Models retrained from saved data.")

# Prediction section
if run_prediction:
    if not prediction_raw_text.strip():
        st.warning("No prediction input was provided, so only the training side remains active.")
    else:
        pred_hash = stable_hash(prediction_raw_text)
        st.session_state["last_predict_hash"] = pred_hash
        pred_df, pred_warnings = parse_prediction_input(prediction_raw_text)
        for msg in pred_warnings:
            st.warning(msg)
        if pred_df.empty:
            st.error("No valid fixture blocks were found in the prediction input.")
        else:
            master_df = read_master()
            standings_df = read_standings()
            bundle = load_model_bundle()
            if bundle is None:
                features_df = read_features()
                bundle, train_warnings = train_models(features_df, master_df)
                for msg in train_warnings:
                    st.warning(msg)
            if bundle is None:
                st.error("The model is not ready yet. Add more weekly results first, then retrain.")
            else:
                predictions = generate_predictions(pred_df, master_df, standings_df, bundle)
                render_prediction_cards(predictions)

# Downloads
master_download = MASTER_PATH.read_bytes() if MASTER_PATH.exists() else b""
features_download = FEATURES_PATH.read_bytes() if FEATURES_PATH.exists() else b""
standings_download = STANDINGS_PATH.read_bytes() if STANDINGS_PATH.exists() else b""
reqs_bytes = b"streamlit\npandas\nnumpy\nscikit-learn\njoblib\n"

st.markdown('<div class="main-card"><div class="section-title">Downloads</div><div class="caption-small">Use these files for auditing, backup, or external modeling work.</div></div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button("Download matches_master.csv", data=master_download, file_name="matches_master.csv", mime="text/csv", use_container_width=True, disabled=not bool(master_download))
with d2:
    st.download_button("Download matches_features.csv", data=features_download, file_name="matches_features.csv", mime="text/csv", use_container_width=True, disabled=not bool(features_download))
with d3:
    st.download_button("Download standings_history.csv", data=standings_download, file_name="standings_history.csv", mime="text/csv", use_container_width=True, disabled=not bool(standings_download))
with d4:
    st.download_button("Download requirements.txt", data=reqs_bytes, file_name="requirements.txt", mime="text/plain", use_container_width=True)
