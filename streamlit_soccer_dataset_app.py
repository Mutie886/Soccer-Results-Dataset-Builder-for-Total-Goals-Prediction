import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "week_sort_key",
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


def safe_float_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def read_master() -> pd.DataFrame:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH)
        missing = [c for c in MASTER_COLUMNS if c not in df.columns]
        for col in missing:
            df[col] = np.nan
        return df[MASTER_COLUMNS]
    return pd.DataFrame(columns=MASTER_COLUMNS)


def read_features() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return pd.DataFrame()


def detect_sequence_direction(weeks_in_appearance_order: List[int]) -> str:
    if len(weeks_in_appearance_order) < 2:
        return "descending"
    descending_moves = 0
    ascending_moves = 0
    for prev, curr in zip(weeks_in_appearance_order[:-1], weeks_in_appearance_order[1:]):
        if curr < prev:
            descending_moves += 1
        elif curr > prev:
            ascending_moves += 1
    return "descending" if descending_moves >= ascending_moves else "ascending"


def parse_sections(raw_text: str, manual_week_number: Optional[int]) -> Tuple[List[Dict], List[str]]:
    warnings: List[str] = []
    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]

    week_header_pattern = re.compile(r"week\s*(\d{1,2})", re.IGNORECASE)
    time_pattern = re.compile(r"\d{1,2}:\d{2}\s*(am|pm)", re.IGNORECASE)

    sections: List[Dict] = []
    current_section: Optional[Dict] = None
    ignored_noise = 0

    for line in raw_lines:
        header_match = week_header_pattern.search(line)
        if header_match:
            week_number = int(header_match.group(1))
            current_section = {
                "week_number": week_number,
                "raw_header": line,
                "lines": [],
            }
            sections.append(current_section)
            continue

        if time_pattern.fullmatch(line):
            ignored_noise += 1
            continue

        if current_section is None:
            current_section = {
                "week_number": manual_week_number,
                "raw_header": None,
                "lines": [],
            }
            sections.append(current_section)

        current_section["lines"].append(line)

    if ignored_noise:
        warnings.append(f"Removed {ignored_noise} timestamp or non-match noise line(s) during cleaning.")

    if not sections:
        return [], ["No usable lines were found after cleaning."]

    if any(sec["week_number"] is None for sec in sections):
        if manual_week_number is None:
            return [], [
                "The pasted input has no week header like 'WEEK 4'. Please provide a manual week number for header-free input."
            ]
        warnings.append(f"No week headers were found for part of the input, so manual week {manual_week_number} was used.")

    return sections, warnings


def sections_to_matches(sections: List[Dict], source_batch_id: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    records: List[Dict] = []

    week_sequence = [int(sec["week_number"]) for sec in sections if sec["week_number"] is not None]
    direction = detect_sequence_direction(week_sequence)

    if direction == "descending":
        reverse_segment = 1
        previous_week = None
        for idx, sec in enumerate(sections):
            current_week = int(sec["week_number"])
            if previous_week is not None and current_week > previous_week:
                reverse_segment += 1
            sec["reverse_segment"] = reverse_segment
            sec["section_index_appearance"] = idx
            previous_week = current_week
        max_segment = max(sec["reverse_segment"] for sec in sections)
        for sec in sections:
            sec["cycle_rank_in_batch"] = max_segment - sec["reverse_segment"] + 1
    else:
        forward_segment = 1
        previous_week = None
        for idx, sec in enumerate(sections):
            current_week = int(sec["week_number"])
            if previous_week is not None and current_week < previous_week:
                forward_segment += 1
            sec["forward_segment"] = forward_segment
            sec["section_index_appearance"] = idx
            sec["cycle_rank_in_batch"] = forward_segment
            previous_week = current_week

    warnings.append(
        f"Detected week sequence as {direction} order and prepared chronological week blocks for storage."
    )

    for sec in sections:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in sec["lines"] if ln.strip()]
        remainder = len(lines) % 4
        if remainder:
            warnings.append(
                f"Week {sec['week_number']} has {len(lines)} usable lines, so the last {remainder} line(s) were ignored."
            )
            lines = lines[: len(lines) - remainder]

        for block_idx in range(0, len(lines), 4):
            home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = lines[block_idx : block_idx + 4]
            home_team = normalize_team_name(home_team_raw)
            away_team = normalize_team_name(away_team_raw)

            if home_team == away_team:
                warnings.append(
                    f"Skipped one block in week {sec['week_number']}: home team and away team are the same ({home_team})."
                )
                continue

            try:
                home_goals = int(home_goals_raw)
                away_goals = int(away_goals_raw)
            except ValueError:
                warnings.append(
                    f"Skipped one block in week {sec['week_number']}: scores must be integers, got '{home_goals_raw}' and '{away_goals_raw}'."
                )
                continue

            if home_goals < 0 or away_goals < 0:
                warnings.append(f"Skipped one block in week {sec['week_number']}: negative goals are not allowed.")
                continue

            total_goals = home_goals + away_goals
            records.append(
                {
                    "source_batch_id": source_batch_id,
                    "batch_cycle_rank": int(sec["cycle_rank_in_batch"]),
                    "week_number": int(sec["week_number"]),
                    "home_team": home_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "away_team": away_team,
                    "total_goals": total_goals,
                    "result": result_code(home_goals, away_goals),
                    "goal_diff": home_goals - away_goals,
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df, warnings

    before = len(df)
    df = df.drop_duplicates(subset=["batch_cycle_rank", "week_number", "home_team", "away_team"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        warnings.append(f"Removed {removed} duplicate match(es) inside the pasted batch.")

    return df, warnings


def map_batch_cycles_to_actual_cycles(batch_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    if batch_df.empty:
        return batch_df

    batch = batch_df.copy()
    batch_cycle_ranks = sorted(batch["batch_cycle_rank"].unique())
    n_batch_cycles = len(batch_cycle_ranks)

    if master_df.empty:
        newest_actual_cycle = n_batch_cycles
    else:
        latest_cycle = int(pd.to_numeric(master_df["cycle_id"], errors="coerce").max())
        latest_cycle_weeks = set(
            pd.to_numeric(master_df.loc[master_df["cycle_id"] == latest_cycle, "week_number"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        latest_week_max = max(latest_cycle_weeks) if latest_cycle_weeks else 0

        newest_batch_rank = max(batch_cycle_ranks)
        newest_batch_weeks = set(batch.loc[batch["batch_cycle_rank"] == newest_batch_rank, "week_number"].astype(int).tolist())
        newest_batch_max = max(newest_batch_weeks)
        overlap = bool(latest_cycle_weeks.intersection(newest_batch_weeks))

        starts_new_cycle = latest_week_max >= 30 and newest_batch_max <= 8 and not overlap
        newest_actual_cycle = latest_cycle + 1 if starts_new_cycle else latest_cycle
        newest_actual_cycle = max(newest_actual_cycle, n_batch_cycles)

    cycle_map = {}
    for offset, batch_rank in enumerate(sorted(batch_cycle_ranks, reverse=True)):
        cycle_map[batch_rank] = newest_actual_cycle - offset

    batch["cycle_id"] = batch["batch_cycle_rank"].map(cycle_map).astype(int)
    batch = batch.drop(columns=["batch_cycle_rank"])
    batch["week_sort_key"] = batch["cycle_id"] * 100 + batch["week_number"]
    batch["match_key"] = (
        batch["cycle_id"].astype(str)
        + "|"
        + batch["week_number"].astype(str)
        + "|"
        + batch["home_team"].astype(str)
        + "|"
        + batch["away_team"].astype(str)
    )
    return batch


def rate_from_series(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == value).mean())


def over_rate(series: pd.Series, threshold: float) -> float:
    if len(series) == 0:
        return np.nan
    return float((series > threshold).mean())


def compute_team_history(prior_matches: pd.DataFrame, team: str) -> pd.DataFrame:
    home_part = prior_matches[prior_matches["home_team"] == team].copy()
    if not home_part.empty:
        home_part["goals_for"] = home_part["home_goals"]
        home_part["goals_against"] = home_part["away_goals"]
        home_part["team_result"] = np.where(
            home_part["home_goals"] > home_part["away_goals"],
            "W",
            np.where(home_part["home_goals"] < home_part["away_goals"], "L", "D"),
        )
        home_part["venue"] = "home"

    away_part = prior_matches[prior_matches["away_team"] == team].copy()
    if not away_part.empty:
        away_part["goals_for"] = away_part["away_goals"]
        away_part["goals_against"] = away_part["home_goals"]
        away_part["team_result"] = np.where(
            away_part["away_goals"] > away_part["home_goals"],
            "W",
            np.where(away_part["away_goals"] < away_part["home_goals"], "L", "D"),
        )
        away_part["venue"] = "away"

    hist = pd.concat([home_part, away_part], ignore_index=True)
    if hist.empty:
        return hist
    return hist.sort_values(["cycle_id", "week_number", "match_id"]).reset_index(drop=True)


def summary_features_from_history(history: pd.DataFrame, prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    def add_window_features(window_name: str, hist_slice: pd.DataFrame) -> None:
        feats[f"{prefix}_{window_name}_avg_scored"] = safe_float_mean(hist_slice["goals_for"])
        feats[f"{prefix}_{window_name}_avg_conceded"] = safe_float_mean(hist_slice["goals_against"])
        feats[f"{prefix}_{window_name}_avg_total_goals"] = safe_float_mean(hist_slice["total_goals"])
        feats[f"{prefix}_{window_name}_win_rate"] = rate_from_series(hist_slice["team_result"], "W")
        feats[f"{prefix}_{window_name}_draw_rate"] = rate_from_series(hist_slice["team_result"], "D")
        feats[f"{prefix}_{window_name}_loss_rate"] = rate_from_series(hist_slice["team_result"], "L")
        feats[f"{prefix}_{window_name}_over_1_5_rate"] = over_rate(hist_slice["total_goals"], 1.5)
        feats[f"{prefix}_{window_name}_over_2_5_rate"] = over_rate(hist_slice["total_goals"], 2.5)
        feats[f"{prefix}_{window_name}_over_3_5_rate"] = over_rate(hist_slice["total_goals"], 3.5)

    add_window_features("last3", history.tail(3))
    add_window_features("last5", history.tail(5))
    add_window_features("last10", history.tail(10))

    home_only = history[history["venue"] == "home"].tail(5)
    away_only = history[history["venue"] == "away"].tail(5)
    feats[f"{prefix}_home_last5_avg_scored"] = safe_float_mean(home_only["goals_for"])
    feats[f"{prefix}_home_last5_avg_conceded"] = safe_float_mean(home_only["goals_against"])
    feats[f"{prefix}_away_last5_avg_scored"] = safe_float_mean(away_only["goals_for"])
    feats[f"{prefix}_away_last5_avg_conceded"] = safe_float_mean(away_only["goals_against"])
    feats[f"{prefix}_matches_played"] = int(len(history))
    return feats


def head_to_head_features(prior_matches: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, float]:
    h2h = prior_matches[
        ((prior_matches["home_team"] == home_team) & (prior_matches["away_team"] == away_team))
        | ((prior_matches["home_team"] == away_team) & (prior_matches["away_team"] == home_team))
    ].copy()
    h2h = h2h.sort_values(["cycle_id", "week_number", "match_id"]).tail(3)

    if h2h.empty:
        return {
            "h2h_last3_avg_total_goals": np.nan,
            "h2h_home_team_win_rate": np.nan,
            "h2h_draw_rate": np.nan,
            "h2h_away_team_win_rate": np.nan,
            "h2h_matches_played": 0,
        }

    normalized_outcomes = []
    for _, match in h2h.iterrows():
        if match["home_team"] == home_team and match["away_team"] == away_team:
            normalized_outcomes.append(match["result"])
        else:
            if match["result"] == "H":
                normalized_outcomes.append("A")
            elif match["result"] == "A":
                normalized_outcomes.append("H")
            else:
                normalized_outcomes.append("D")

    outcomes = pd.Series(normalized_outcomes)
    return {
        "h2h_last3_avg_total_goals": float(h2h["total_goals"].mean()),
        "h2h_home_team_win_rate": rate_from_series(outcomes, "H"),
        "h2h_draw_rate": rate_from_series(outcomes, "D"),
        "h2h_away_team_win_rate": rate_from_series(outcomes, "A"),
        "h2h_matches_played": int(len(h2h)),
    }


def build_feature_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()
    df["cycle_id"] = pd.to_numeric(df["cycle_id"], errors="coerce").astype(int)
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype(int)
    df = df.sort_values(["cycle_id", "week_number", "match_id"]).reset_index(drop=True)

    rows: List[Dict] = []
    week_groups = list(df.groupby(["cycle_id", "week_number"], sort=True))

    for cycle_week, week_df in week_groups:
        cycle_id, week_number = cycle_week
        prior = df[(df["cycle_id"] < cycle_id) | ((df["cycle_id"] == cycle_id) & (df["week_number"] < week_number))].copy()

        for _, row in week_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            home_hist = compute_team_history(prior, home_team)
            away_hist = compute_team_history(prior, away_team)

            features = {
                "match_id": int(row["match_id"]),
                "cycle_id": int(cycle_id),
                "week_number": int(week_number),
                "home_team": home_team,
                "away_team": away_team,
            }
            features.update(summary_features_from_history(home_hist, "home_team"))
            features.update(summary_features_from_history(away_hist, "away_team"))
            features.update(head_to_head_features(prior, home_team, away_team))
            features["history_pool_matches"] = int(len(prior))
            features["target_total_goals"] = int(row["total_goals"])
            features["target_total_class"] = total_goal_class(int(row["total_goals"]))
            rows.append(features)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df

    feature_cols = [c for c in feat_df.columns if c not in {"match_id", "cycle_id", "week_number", "home_team", "away_team", "target_total_goals", "target_total_class"}]
    for col in feature_cols:
        if feat_df[col].isna().any():
            feat_df[f"{col}_missing"] = feat_df[col].isna().astype(int)
            feat_df[col] = feat_df[col].fillna(0.0)

    return feat_df


def append_to_master(new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    master = read_master()
    staged = map_batch_cycles_to_actual_cycles(new_df, master)

    existing_keys = set(master["match_key"].astype(str)) if not master.empty else set()
    is_duplicate_existing = staged["match_key"].astype(str).isin(existing_keys)
    rejected_existing = staged[is_duplicate_existing].copy()
    accepted = staged[~is_duplicate_existing].copy()

    if not accepted.empty:
        start_id = 1 if master.empty else int(pd.to_numeric(master["match_id"], errors="coerce").max()) + 1
        accepted = accepted.reset_index(drop=True)
        accepted.insert(0, "match_id", range(start_id, start_id + len(accepted)))
        master = pd.concat([master, accepted[MASTER_COLUMNS]], ignore_index=True)
        master = master.sort_values(["cycle_id", "week_number", "match_id"]).reset_index(drop=True)
        master.to_csv(MASTER_PATH, index=False)

    if not rejected_existing.empty:
        rejected_existing.to_csv(REJECTED_PATH, mode="a", header=not REJECTED_PATH.exists(), index=False)

    features = build_feature_dataset(master)
    if not features.empty:
        features.to_csv(FEATURES_PATH, index=False)
    elif FEATURES_PATH.exists():
        FEATURES_PATH.unlink()

    return master, rejected_existing, len(accepted), features


# ------------------------------ UI ------------------------------
st.title("⚽ Soccer Results Dataset Builder")
st.caption(
    "Clean raw electronic soccer results, infer cycle and week order, store history without duplicates, and generate a model-ready total-goals training dataset."
)

with st.expander("Accepted input styles", expanded=True):
    st.markdown(
        """
**Style 1: with week headers**
```text
English League WEEK 4 - #2026042118
11:08 pm
London Blues
1
0
London Reds
...
```

**Style 2: clean match-only blocks**
```text
Sheffield U
3
0
Southampton
Palace
2
1
Fulham
```

Each match must still resolve to 4 lines:
`home team -> home goals -> away goals -> away team`
        """
    )

col1, col2 = st.columns([2.5, 1])
with col1:
    raw_text = st.text_area(
        "Paste raw results",
        height=320,
        placeholder="Paste week headers, timestamps, or plain 4-line match blocks here...",
    )
with col2:
    manual_week_number = st.number_input(
        "Manual week number",
        min_value=1,
        max_value=38,
        value=1,
        help="Used only when the pasted input has no WEEK header.",
    )
    source_batch_id = st.text_input(
        "Source batch id",
        value=pd.Timestamp.utcnow().strftime("batch_%Y%m%d_%H%M%S"),
    )
    save_button = st.button("Clean, process, and save", type="primary", use_container_width=True)
    reset_button = st.button("Refresh system / start new dataset", type="secondary", use_container_width=True)

notification_messages: List[Tuple[str, str]] = []

if reset_button:
    removed_files = []
    for path in [MASTER_PATH, FEATURES_PATH, REJECTED_PATH]:
        try:
            if path.exists():
                path.unlink()
                removed_files.append(path.name)
        except Exception as exc:
            notification_messages.append(("error", f"Could not remove {path.name}: {exc}"))
    if removed_files:
        notification_messages.append(("success", "System refreshed. Removed saved dataset files: " + ", ".join(removed_files)))
    else:
        notification_messages.append(("info", "System was already empty. No saved dataset files were found."))

if save_button:
    sections, parse_warnings = parse_sections(raw_text, int(manual_week_number) if raw_text.strip() else None)
    notification_messages.extend([("warning", msg) for msg in parse_warnings])

    if sections:
        parsed_df, build_warnings = sections_to_matches(sections, source_batch_id.strip() or "batch_manual")
        notification_messages.extend([("warning", msg) for msg in build_warnings])

        if parsed_df.empty:
            notification_messages.append(("error", "No valid matches were found after cleaning and parsing."))
        else:
            master_df, rejected_existing, accepted_count, features_df = append_to_master(parsed_df)

            if accepted_count > 0:
                notification_messages.append(("success", f"Saved {accepted_count} new match(es) into the system."))
            else:
                notification_messages.append(("info", "All cleaned matches from this batch already exist in the saved history."))

            if not rejected_existing.empty:
                notification_messages.append(("info", f"Ignored {len(rejected_existing)} existing duplicate match(es)."))

for level, message in notification_messages:
    getattr(st, level)(message)

master_saved = read_master()
features_saved = read_features()

st.subheader("Notifications dashboard")
metric_cols = st.columns(5)
master_rows = len(master_saved)
feature_rows = len(features_saved)
teams_seen = 0 if master_saved.empty else len(pd.unique(pd.concat([master_saved["home_team"], master_saved["away_team"]], ignore_index=True)))

cycle_numeric = pd.Series(dtype="float64") if master_saved.empty or "cycle_id" not in master_saved.columns else pd.to_numeric(master_saved["cycle_id"], errors="coerce")
cycle_max = cycle_numeric.max() if not cycle_numeric.empty else float("nan")
cycles_seen = 0 if pd.isna(cycle_max) else int(cycle_max)

if master_saved.empty or "week_number" not in master_saved.columns:
    latest_week = 0
else:
    week_sorted = master_saved.copy()
    if "cycle_id" in week_sorted.columns:
        week_sorted["_cycle_num"] = pd.to_numeric(week_sorted["cycle_id"], errors="coerce")
    else:
        week_sorted["_cycle_num"] = 0
    week_sorted["_week_num"] = pd.to_numeric(week_sorted["week_number"], errors="coerce")
    week_sorted = week_sorted.sort_values(["_cycle_num", "_week_num"], na_position="last")
    latest_week_value = week_sorted["_week_num"].iloc[-1] if not week_sorted.empty else float("nan")
    latest_week = 0 if pd.isna(latest_week_value) else int(latest_week_value)
metric_cols[0].metric("Master rows", master_rows)
metric_cols[1].metric("Feature rows", feature_rows)
metric_cols[2].metric("Teams seen", teams_seen)
metric_cols[3].metric("Cycles seen", cycles_seen)
metric_cols[4].metric("Latest week", latest_week)

if master_saved.empty:
    st.info("No saved records yet. Paste a batch above to create the datasets.")
else:
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download matches_master.csv",
            data=master_saved.to_csv(index=False).encode("utf-8"),
            file_name="matches_master.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "Download matches_features.csv",
            data=features_saved.to_csv(index=False).encode("utf-8"),
            file_name="matches_features.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Model-readiness notes"):
        st.markdown(
            """
- Week headers are extracted automatically when present.
- Real dates are not required; the system uses `cycle_id + week_number` for chronology.
- Duplicate protection is based on `cycle_id + week_number + home_team + away_team`.
- Feature rows are built using **earlier weeks only**, so matches from the same week do not leak into each other.
- Missing historical values are filled with `0.0`, and matching `_missing` indicator columns are added for the model.
            """
        )
