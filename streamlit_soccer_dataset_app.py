import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soccer Results Dataset Builder", layout="wide")

DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
MASTER_PATH = DATA_DIR / "matches_master.csv"
FEATURES_PATH = DATA_DIR / "matches_features.csv"
REJECTED_PATH = DATA_DIR / "rejected_duplicates.csv"

MASTER_COLUMNS = [
    "match_id",
    "cycle_id",
    "week_number",
    "section_order",
    "batch_match_number",
    "global_order",
    "source_batch_id",
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

STANDARD_HISTORY_COLS = [
    "match_id",
    "cycle_id",
    "week_number",
    "global_order",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "total_goals",
    "goals_for",
    "goals_against",
    "team_result",
    "venue",
]


# ------------------------------ Helpers ------------------------------
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


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def safe_rate(series: pd.Series, value) -> float:
    return float((series == value).mean()) if len(series) else np.nan


def safe_over_rate(series: pd.Series, threshold: float) -> float:
    return float((series > threshold).mean()) if len(series) else np.nan


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_master() -> pd.DataFrame:
    df = read_csv_safe(MASTER_PATH)
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[MASTER_COLUMNS].copy()


def read_features() -> pd.DataFrame:
    return read_csv_safe(FEATURES_PATH)


def get_last_numeric(df: pd.DataFrame, column: str, default: int = 0) -> int:
    if df.empty or column not in df.columns:
        return default
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    if vals.empty:
        return default
    return int(vals.max())


def clear_saved_data() -> None:
    for path in [MASTER_PATH, FEATURES_PATH, REJECTED_PATH]:
        if path.exists():
            path.unlink()


# ------------------------------ Parsing ------------------------------
def is_noise_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", low):
        return True
    return False


def extract_week_number(line: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", line, flags=re.I)
    if m:
        week = int(m.group(1))
        if 1 <= week <= 38:
            return week
    return None


def parse_matches(raw_text: str, manual_week_number: Optional[int], source_batch_id: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse text into week sections. The text is assumed to be pasted with the
    latest content at the top. For chronological storage, sections are reversed
    and matches inside each section are reversed.
    """
    warnings: List[str] = []

    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]

    if not raw_lines:
        return pd.DataFrame(), ["No input text was provided."]

    sections_encountered = []
    current_week = None
    current_lines: List[str] = []
    section_counter = 0
    removed_noise = 0

    has_header_week = any(extract_week_number(ln) is not None for ln in raw_lines)
    if not has_header_week and manual_week_number is None:
        return pd.DataFrame(), ["No week header was found. Enter a week number manually."]

    if not has_header_week:
        current_week = int(manual_week_number)

    for line in raw_lines:
        week = extract_week_number(line)
        if week is not None:
            if current_lines:
                section_counter += 1
                sections_encountered.append(
                    {"section_order": section_counter, "week_number": current_week, "lines": current_lines.copy()}
                )
                current_lines = []
            current_week = week
            continue

        if is_noise_line(line):
            removed_noise += 1
            continue

        if current_week is None:
            current_week = int(manual_week_number) if manual_week_number is not None else None

        if current_week is None:
            removed_noise += 1
            continue

        current_lines.append(line)

    if current_lines:
        section_counter += 1
        sections_encountered.append(
            {"section_order": section_counter, "week_number": current_week, "lines": current_lines.copy()}
        )

    if removed_noise:
        warnings.append(f"Removed {removed_noise} non-match noise line(s) such as timestamps.")

    if not sections_encountered:
        return pd.DataFrame(), ["No usable match sections were found after cleaning."]

    parsed_records = []
    rejected = []

    # Reverse sections because the top section is the latest in the pasted text.
    chronological_sections = list(reversed(sections_encountered))

    for chrono_section_index, section in enumerate(chronological_sections, start=1):
        lines = section["lines"]
        remainder = len(lines) % 4
        if remainder:
            warnings.append(
                f"Week {section['week_number']} has {len(lines)} usable lines; the last {remainder} line(s) were ignored."
            )
            lines = lines[: len(lines) - remainder]

        matches_in_top_to_bottom = []
        for idx in range(0, len(lines), 4):
            home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = lines[idx : idx + 4]
            home_team = normalize_team_name(home_team_raw)
            away_team = normalize_team_name(away_team_raw)

            if home_team == away_team:
                rejected.append(
                    f"Skipped a block in week {section['week_number']}: home team and away team are the same ({home_team})."
                )
                continue

            try:
                home_goals = int(home_goals_raw)
                away_goals = int(away_goals_raw)
            except ValueError:
                rejected.append(
                    f"Skipped a block in week {section['week_number']}: scores must be integers, got '{home_goals_raw}' and '{away_goals_raw}'."
                )
                continue

            if home_goals < 0 or away_goals < 0:
                rejected.append(f"Skipped a block in week {section['week_number']}: negative goals are not allowed.")
                continue

            matches_in_top_to_bottom.append(
                {
                    "week_number": int(section["week_number"]),
                    "section_order": chrono_section_index,
                    "home_team": home_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "away_team": away_team,
                }
            )

        # Top match is latest, so reverse for chronological storage.
        for batch_match_number, rec in enumerate(reversed(matches_in_top_to_bottom), start=1):
            total_goals = rec["home_goals"] + rec["away_goals"]
            rec.update(
                {
                    "batch_match_number": batch_match_number,
                    "source_batch_id": source_batch_id,
                    "total_goals": total_goals,
                    "result": result_code(rec["home_goals"], rec["away_goals"]),
                    "goal_diff": rec["home_goals"] - rec["away_goals"],
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
            parsed_records.append(rec)

    parsed_df = pd.DataFrame(parsed_records)
    if parsed_df.empty:
        return parsed_df, warnings + rejected

    # Remove duplicates inside this pasted input using week + teams + score.
    before = len(parsed_df)
    parsed_df = parsed_df.drop_duplicates(
        subset=["week_number", "home_team", "away_team", "home_goals", "away_goals"]
    ).reset_index(drop=True)
    removed_dupes = before - len(parsed_df)
    if removed_dupes:
        warnings.append(f"Removed {removed_dupes} duplicate match(es) inside this pasted input.")

    if rejected:
        warnings.extend(rejected)

    return parsed_df, warnings


# ------------------------------ Append & ordering ------------------------------
def assign_cycle_ids_and_keys(existing_master: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if new_df.empty:
        return new_df.copy()

    df = new_df.copy().reset_index(drop=True)

    last_cycle = get_last_numeric(existing_master, "cycle_id", 0)
    last_week = get_last_numeric(existing_master.tail(1), "week_number", 0)
    last_global_order = get_last_numeric(existing_master, "global_order", 0)
    last_match_id = get_last_numeric(existing_master, "match_id", 0)

    cycle_ids = []
    match_keys = []
    global_orders = []
    match_ids = []

    current_cycle = last_cycle if last_cycle > 0 else 1
    prev_week = last_week if last_week > 0 else None

    for i, row in df.iterrows():
        week = int(row["week_number"])

        if prev_week is None:
            # First ever row in the system or first valid row after empty history.
            current_cycle = current_cycle if current_cycle > 0 else 1
        elif week == prev_week:
            pass
        elif prev_week == 38 and week == 1:
            current_cycle += 1
        elif week < prev_week:
            # Conservative fallback: treat a decrease in week order as a new cycle.
            current_cycle += 1

        cycle_ids.append(current_cycle)
        match_keys.append(f"{current_cycle}|{week}|{row['home_team']}|{row['away_team']}")
        global_orders.append(last_global_order + i + 1)
        match_ids.append(last_match_id + i + 1)
        prev_week = week

    df.insert(0, "match_id", match_ids)
    df["cycle_id"] = cycle_ids
    df["global_order"] = global_orders
    df["match_key"] = match_keys

    ordered_cols = [c for c in MASTER_COLUMNS if c in df.columns]
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[MASTER_COLUMNS].copy()


def append_to_master(parsed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    master = read_master()

    assigned = assign_cycle_ids_and_keys(master, parsed_df)
    existing_keys = set(master["match_key"].astype(str)) if not master.empty else set()

    is_duplicate_existing = assigned["match_key"].astype(str).isin(existing_keys)
    rejected_existing = assigned[is_duplicate_existing].copy()
    accepted = assigned[~is_duplicate_existing].copy()

    if not rejected_existing.empty:
        prev = read_csv_safe(REJECTED_PATH)
        rejected_all = pd.concat([prev, rejected_existing], ignore_index=True) if not prev.empty else rejected_existing
        rejected_all.to_csv(REJECTED_PATH, index=False)

    if accepted.empty:
        features = read_features()
        return master, rejected_existing, 0, features

    master = pd.concat([master, accepted], ignore_index=True)
    master = master.sort_values(["cycle_id", "week_number", "section_order", "batch_match_number", "global_order"]).reset_index(drop=True)
    master.to_csv(MASTER_PATH, index=False)

    features = build_feature_dataset(master)
    features.to_csv(FEATURES_PATH, index=False)
    return master, rejected_existing, len(accepted), features


# ------------------------------ Feature engineering ------------------------------
def compute_team_history(prior_matches: pd.DataFrame, team: str, venue: Optional[str] = None) -> pd.DataFrame:
    if prior_matches.empty:
        return pd.DataFrame(columns=STANDARD_HISTORY_COLS)

    home = prior_matches[prior_matches["home_team"] == team].copy()
    if not home.empty:
        home["goals_for"] = home["home_goals"]
        home["goals_against"] = home["away_goals"]
        home["team_result"] = np.where(home["home_goals"] > home["away_goals"], "W", np.where(home["home_goals"] < home["away_goals"], "L", "D"))
        home["venue"] = "home"

    away = prior_matches[prior_matches["away_team"] == team].copy()
    if not away.empty:
        away["goals_for"] = away["away_goals"]
        away["goals_against"] = away["home_goals"]
        away["team_result"] = np.where(away["away_goals"] > away["home_goals"], "W", np.where(away["away_goals"] < away["home_goals"], "L", "D"))
        away["venue"] = "away"

    hist = pd.concat([home, away], ignore_index=True) if (not home.empty or not away.empty) else pd.DataFrame(columns=STANDARD_HISTORY_COLS)
    for col in STANDARD_HISTORY_COLS:
        if col not in hist.columns:
            hist[col] = pd.Series(dtype="float64")
    hist = hist[STANDARD_HISTORY_COLS].sort_values(["cycle_id", "week_number", "global_order"]).reset_index(drop=True)
    if venue is not None:
        hist = hist[hist["venue"] == venue].reset_index(drop=True)
    return hist


def summary_features_from_history(history: pd.DataFrame, prefix: str) -> dict:
    feats = {}

    def add_window(name: str, hist_slice: pd.DataFrame) -> None:
        feats[f"{prefix}_{name}_avg_scored"] = safe_mean(hist_slice["goals_for"]) if "goals_for" in hist_slice else np.nan
        feats[f"{prefix}_{name}_avg_conceded"] = safe_mean(hist_slice["goals_against"]) if "goals_against" in hist_slice else np.nan
        feats[f"{prefix}_{name}_avg_total_goals"] = safe_mean(hist_slice["total_goals"]) if "total_goals" in hist_slice else np.nan
        feats[f"{prefix}_{name}_win_rate"] = safe_rate(hist_slice["team_result"], "W") if "team_result" in hist_slice else np.nan
        feats[f"{prefix}_{name}_draw_rate"] = safe_rate(hist_slice["team_result"], "D") if "team_result" in hist_slice else np.nan
        feats[f"{prefix}_{name}_loss_rate"] = safe_rate(hist_slice["team_result"], "L") if "team_result" in hist_slice else np.nan
        feats[f"{prefix}_{name}_over_1_5_rate"] = safe_over_rate(hist_slice["total_goals"], 1.5) if "total_goals" in hist_slice else np.nan
        feats[f"{prefix}_{name}_over_2_5_rate"] = safe_over_rate(hist_slice["total_goals"], 2.5) if "total_goals" in hist_slice else np.nan
        feats[f"{prefix}_{name}_over_3_5_rate"] = safe_over_rate(hist_slice["total_goals"], 3.5) if "total_goals" in hist_slice else np.nan

    add_window("last3", history.tail(3))
    add_window("last5", history.tail(5))
    add_window("last10", history.tail(10))

    home_only = history[history["venue"] == "home"].tail(5) if "venue" in history else pd.DataFrame()
    away_only = history[history["venue"] == "away"].tail(5) if "venue" in history else pd.DataFrame()

    feats[f"{prefix}_home_last5_avg_scored"] = safe_mean(home_only["goals_for"]) if len(home_only) else np.nan
    feats[f"{prefix}_home_last5_avg_conceded"] = safe_mean(home_only["goals_against"]) if len(home_only) else np.nan
    feats[f"{prefix}_away_last5_avg_scored"] = safe_mean(away_only["goals_for"]) if len(away_only) else np.nan
    feats[f"{prefix}_away_last5_avg_conceded"] = safe_mean(away_only["goals_against"]) if len(away_only) else np.nan
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
        ((prior_matches["home_team"] == home_team) & (prior_matches["away_team"] == away_team))
        | ((prior_matches["home_team"] == away_team) & (prior_matches["away_team"] == home_team))
    ].copy()
    h2h = h2h.sort_values(["cycle_id", "week_number", "global_order"]).tail(3)

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

    outcomes = pd.Series(outcomes)
    return {
        "h2h_last3_avg_total_goals": float(h2h["total_goals"].mean()),
        "h2h_home_team_win_rate": safe_rate(outcomes, "H"),
        "h2h_draw_rate": safe_rate(outcomes, "D"),
        "h2h_away_team_win_rate": safe_rate(outcomes, "A"),
        "h2h_matches_played": int(len(h2h)),
    }


def build_feature_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()
    for col in ["cycle_id", "week_number", "global_order", "match_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["cycle_id", "week_number", "section_order", "batch_match_number", "global_order"]).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        # Only earlier weeks are allowed; same-week matches are excluded to avoid leakage.
        prior = df[
            (df["cycle_id"] < row["cycle_id"])
            | ((df["cycle_id"] == row["cycle_id"]) & (df["week_number"] < row["week_number"]))
        ].copy()

        home_team = row["home_team"]
        away_team = row["away_team"]
        home_hist = compute_team_history(prior, home_team)
        away_hist = compute_team_history(prior, away_team)

        features = {
            "match_id": int(row["match_id"]),
            "cycle_id": int(row["cycle_id"]),
            "week_number": int(row["week_number"]),
            "global_order": int(row["global_order"]),
            "home_team": home_team,
            "away_team": away_team,
        }
        features.update(summary_features_from_history(home_hist, "home_team"))
        features.update(summary_features_from_history(away_hist, "away_team"))
        features.update(head_to_head_features(prior, home_team, away_team))
        features["target_total_goals"] = int(row["total_goals"])
        features["target_total_class"] = total_goal_class(int(row["total_goals"]))
        rows.append(features)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df

    feature_cols = [c for c in feat_df.columns if c not in {"match_id", "cycle_id", "week_number", "global_order", "home_team", "away_team", "target_total_goals", "target_total_class"}]
    for col in feature_cols:
        missing_col = f"{col}_missing"
        feat_df[missing_col] = feat_df[col].isna().astype(int)
        feat_df[col] = feat_df[col].fillna(0.0)

    return feat_df


# ------------------------------ UI ------------------------------
st.title("⚽ Soccer Results Dataset Builder")
st.caption("Clean raw electronic soccer results, store them in chronological cycle/week order, and download model-ready datasets.")

with st.expander("Expected raw format", expanded=False):
    st.code(
        """English League WEEK 5 - #2026042117
9:54 pm
Sheffield U
3
0
Southampton
Palace
2
1
Fulham""",
        language="text",
    )
    st.write("Headers and time lines are cleaned automatically. If there is no week header in the pasted text, enter the week number manually.")

left, right = st.columns([3, 1])
with left:
    raw_text = st.text_area("Paste raw match input", height=320, placeholder="Paste match text here...")
with right:
    manual_week = st.number_input("Week number (only if no week header exists)", min_value=1, max_value=38, value=1)
    source_batch_id = st.text_input("Source batch id", value=pd.Timestamp.utcnow().strftime("batch_%Y%m%d_%H%M%S"))
    save_button = st.button("Process and save", type="primary", use_container_width=True)
    reset_button = st.button("Refresh system / start new dataset", use_container_width=True)

if reset_button:
    clear_saved_data()
    st.success("The saved system records were cleared. You can now build a new dataset from scratch.")

master_saved = read_master()
features_saved = read_features()

st.subheader("Notifications dashboard")
metric_cols = st.columns(5)
metric_cols[0].metric("Saved matches", int(len(master_saved)))
metric_cols[1].metric("Feature rows", int(len(features_saved)))
metric_cols[2].metric("Teams seen", int(len(pd.unique(pd.concat([master_saved["home_team"], master_saved["away_team"]], ignore_index=True))) if not master_saved.empty else 0))
metric_cols[3].metric("Current cycle", get_last_numeric(master_saved, "cycle_id", 0))
metric_cols[4].metric("Latest week", get_last_numeric(master_saved.tail(1), "week_number", 0))

if save_button:
    parsed_df, warnings = parse_matches(raw_text, int(manual_week), source_batch_id.strip() or "batch_manual")

    for msg in warnings:
        st.warning(msg)

    if parsed_df.empty:
        st.error("No valid matches were found after cleaning.")
    else:
        master_df, rejected_existing, accepted_count, features_df = append_to_master(parsed_df)

        if accepted_count > 0:
            st.success(f"Saved {accepted_count} new match(es).")
        else:
            st.info("No new matches were added because all cleaned matches already existed in the saved history.")

        if not rejected_existing.empty:
            st.info(f"Ignored {len(rejected_existing)} existing duplicate match(es) already in the saved system.")

        st.subheader("Notifications dashboard")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Saved matches", int(len(master_df)))
        metric_cols[1].metric("Feature rows", int(len(features_df)))
        metric_cols[2].metric("Teams seen", int(len(pd.unique(pd.concat([master_df["home_team"], master_df["away_team"]], ignore_index=True)))))
        metric_cols[3].metric("Current cycle", get_last_numeric(master_df, "cycle_id", 0))
        metric_cols[4].metric("Latest week", get_last_numeric(master_df.tail(1), "week_number", 0))

        if not features_df.empty:
            missing_indicators = [c for c in features_df.columns if c.endswith("_missing")]
            total_missing_flags = int(features_df[missing_indicators].sum().sum()) if missing_indicators else 0
            st.success(
                f"Model-ready dataset updated. Missing historical values were filled with 0.0 and tracked with missing-indicator columns. Total missing flags: {total_missing_flags}."
            )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download master dataset CSV",
                data=master_df.to_csv(index=False).encode("utf-8"),
                file_name="matches_master.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download model-ready feature dataset CSV",
                data=features_df.to_csv(index=False).encode("utf-8"),
                file_name="matches_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

if not master_saved.empty and not features_saved.empty:
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download saved master dataset CSV",
            data=master_saved.to_csv(index=False).encode("utf-8"),
            file_name="matches_master.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_saved_master",
        )
    with dl2:
        st.download_button(
            "Download saved model-ready feature dataset CSV",
            data=features_saved.to_csv(index=False).encode("utf-8"),
            file_name="matches_features.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_saved_features",
        )
