import re
from pathlib import Path
from typing import List, Tuple

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
    "match_date",
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
    """Clean team names while keeping the user's naming convention."""
    name = re.sub(r"\s+", " ", str(name).strip())
    return name.title()



def read_master() -> pd.DataFrame:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH)
        # Keep column order stable if file already exists.
        missing = [c for c in MASTER_COLUMNS if c not in df.columns]
        for col in missing:
            df[col] = np.nan
        return df[MASTER_COLUMNS]
    return pd.DataFrame(columns=MASTER_COLUMNS)



def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"



def total_goal_class(total_goals: int) -> str:
    return str(total_goals) if total_goals <= 6 else "6_plus"



def parse_matches(raw_text: str, match_date: str, source_batch_id: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse raw text in repeated blocks of 4 cleaned lines:
    home_team, home_goals, away_goals, away_team.

    The cleaner removes noise lines such as league/week headers,
    timestamps, batch ids, and other non-match metadata before grouping.
    """
    warnings: List[str] = []

    def is_noise_line(line: str) -> bool:
        low = line.lower().strip()
        if not low:
            return True
        # time stamps like 11:08 pm / 9:54 am
        if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", low):
            return True
        # batch / competition headers like English League WEEK 4 - #2026042118
        if "week" in low and "#" in low:
            return True
        if low.startswith("english league"):
            return True
        if low.startswith("league "):
            return True
        return False

    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines()]
    removed_noise = [ln for ln in raw_lines if ln and is_noise_line(ln)]
    lines = [ln for ln in raw_lines if ln and not is_noise_line(ln)]

    if removed_noise:
        warnings.append(f"Removed {len(removed_noise)} non-match noise line(s) such as headers or timestamps before parsing.")

    if not lines:
        return pd.DataFrame(), ["No usable lines were found after cleaning."]

    remainder = len(lines) % 4
    if remainder != 0:
        warnings.append(
            f"The cleaned input has {len(lines)} usable lines, which is not divisible by 4. "
            f"The last {remainder} line(s) will be ignored."
        )
        lines = lines[: len(lines) - remainder]

    records = []
    rejected = []

    for idx in range(0, len(lines), 4):
        home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = lines[idx : idx + 4]

        home_team = normalize_team_name(home_team_raw)
        away_team = normalize_team_name(away_team_raw)

        if home_team == away_team:
            rejected.append(
                f"Skipped block {idx // 4 + 1}: home team and away team are the same ({home_team})."
            )
            continue

        try:
            home_goals = int(home_goals_raw)
            away_goals = int(away_goals_raw)
        except ValueError:
            rejected.append(
                f"Skipped block {idx // 4 + 1}: scores must be integers, got '{home_goals_raw}' and '{away_goals_raw}'."
            )
            continue

        if home_goals < 0 or away_goals < 0:
            rejected.append(
                f"Skipped block {idx // 4 + 1}: negative goals are not allowed."
            )
            continue

        total_goals = home_goals + away_goals
        gd = home_goals - away_goals
        key = f"{match_date}|{home_team}|{away_team}|{home_goals}|{away_goals}"

        records.append(
            {
                "match_date": match_date,
                "source_batch_id": source_batch_id,
                "home_team": home_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "away_team": away_team,
                "total_goals": total_goals,
                "result": result_code(home_goals, away_goals),
                "goal_diff": gd,
                "match_key": key,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
        )

    if rejected:
        warnings.extend(rejected)

    df = pd.DataFrame(records)
    if df.empty:
        return df, warnings

    before = len(df)
    df = df.drop_duplicates(subset=["match_key"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        warnings.append(f"Removed {removed} duplicate match(es) inside the pasted batch.")

    return df, warnings



def compute_team_history(prior_matches: pd.DataFrame, team: str, venue: str | None = None) -> pd.DataFrame:
    """Return prior matches from the team perspective with goals_for/goals_against columns."""
    home_part = prior_matches[prior_matches["home_team"] == team].copy()
    home_part["goals_for"] = home_part["home_goals"]
    home_part["goals_against"] = home_part["away_goals"]
    home_part["team_result"] = np.where(
        home_part["home_goals"] > home_part["away_goals"],
        "W",
        np.where(home_part["home_goals"] < home_part["away_goals"], "L", "D"),
    )
    home_part["venue"] = "home"

    away_part = prior_matches[prior_matches["away_team"] == team].copy()
    away_part["goals_for"] = away_part["away_goals"]
    away_part["goals_against"] = away_part["home_goals"]
    away_part["team_result"] = np.where(
        away_part["away_goals"] > away_part["home_goals"],
        "W",
        np.where(away_part["away_goals"] < away_part["home_goals"], "L", "D"),
    )
    away_part["venue"] = "away"

    hist = pd.concat([home_part, away_part], ignore_index=True)
    hist = hist.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    if venue is not None:
        hist = hist[hist["venue"] == venue].reset_index(drop=True)
    return hist



def rate_from_series(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == value).mean())



def over_rate(series: pd.Series, threshold: float) -> float:
    if len(series) == 0:
        return np.nan
    return float((series > threshold).mean())



def summary_features_from_history(history: pd.DataFrame, prefix: str) -> dict:
    """Rolling features using past matches only."""
    feats: dict = {}

    def add_window_features(window_name: str, hist_slice: pd.DataFrame) -> None:
        feats[f"{prefix}_{window_name}_avg_scored"] = hist_slice["goals_for"].mean() if len(hist_slice) else np.nan
        feats[f"{prefix}_{window_name}_avg_conceded"] = hist_slice["goals_against"].mean() if len(hist_slice) else np.nan
        feats[f"{prefix}_{window_name}_avg_total_goals"] = hist_slice["total_goals"].mean() if len(hist_slice) else np.nan
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

    feats[f"{prefix}_home_last5_avg_scored"] = home_only["goals_for"].mean() if len(home_only) else np.nan
    feats[f"{prefix}_home_last5_avg_conceded"] = home_only["goals_against"].mean() if len(home_only) else np.nan
    feats[f"{prefix}_away_last5_avg_scored"] = away_only["goals_for"].mean() if len(away_only) else np.nan
    feats[f"{prefix}_away_last5_avg_conceded"] = away_only["goals_against"].mean() if len(away_only) else np.nan
    feats[f"{prefix}_matches_played"] = int(len(history))
    return feats



def head_to_head_features(prior_matches: pd.DataFrame, home_team: str, away_team: str) -> dict:
    h2h = prior_matches[
        ((prior_matches["home_team"] == home_team) & (prior_matches["away_team"] == away_team))
        | ((prior_matches["home_team"] == away_team) & (prior_matches["away_team"] == home_team))
    ].copy()
    h2h = h2h.sort_values(["match_date", "match_id"]).tail(3)

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
        else:
            if row["home_team"] == home_team:
                outcomes.append("H" if row["home_goals"] > row["away_goals"] else "A")
            else:
                outcomes.append("H" if row["away_goals"] > row["home_goals"] else "A")

    outcomes = pd.Series(outcomes)
    return {
        "h2h_last3_avg_total_goals": h2h["total_goals"].mean(),
        "h2h_home_team_win_rate": rate_from_series(outcomes, "H"),
        "h2h_draw_rate": rate_from_series(outcomes, "D"),
        "h2h_away_team_win_rate": rate_from_series(outcomes, "A"),
        "h2h_matches_played": int(len(h2h)),
    }



def build_feature_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        prior = df.iloc[:idx].copy()
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_hist = compute_team_history(prior, home_team)
        away_hist = compute_team_history(prior, away_team)

        features = {
            "match_id": row["match_id"],
            "match_date": row["match_date"].date().isoformat(),
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
    return feat_df



def append_to_master(new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    master = read_master()

    if master.empty:
        existing_keys = set()
    else:
        existing_keys = set(master["match_key"].astype(str))

    is_duplicate_existing = new_df["match_key"].astype(str).isin(existing_keys)
    rejected_existing = new_df[is_duplicate_existing].copy()
    accepted = new_df[~is_duplicate_existing].copy()

    if not accepted.empty:
        start_id = 1 if master.empty else int(pd.to_numeric(master["match_id"], errors="coerce").max()) + 1
        accepted = accepted.reset_index(drop=True)
        accepted.insert(0, "match_id", range(start_id, start_id + len(accepted)))
        master = pd.concat([master, accepted[MASTER_COLUMNS]], ignore_index=True)
        master = master.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        master.to_csv(MASTER_PATH, index=False)

    if not rejected_existing.empty:
        rejected_existing.to_csv(REJECTED_PATH, mode="a", header=not REJECTED_PATH.exists(), index=False)

    features = build_feature_dataset(master)
    if not features.empty:
        features.to_csv(FEATURES_PATH, index=False)

    return master, rejected_existing, len(accepted)


# ------------------------------ UI ------------------------------
st.title("⚽ Soccer Results Dataset Builder")
st.caption(
    "Paste raw match results, clean them automatically, remove duplicates, append to history, and build a feature dataset for later total-goals prediction."
)

with st.expander("Expected raw input format", expanded=True):
    st.code(
        """London Reds
3
1
Manchester Reds
Tottenham
2
1
Sheffield U""",
        language="text",
    )
    st.write(
        "Each match must appear as 4 lines: home team, home goals, away goals, away team. "
        "Blank lines are allowed because the app removes them during cleaning."
    )

col1, col2 = st.columns([2, 1])
with col1:
    raw_text = st.text_area(
        "Paste raw match results",
        height=320,
        placeholder="Paste the uncleaned match text here...",
    )
with col2:
    match_date = st.date_input("Match date")
    source_batch_id = st.text_input("Source batch id", value=pd.Timestamp.utcnow().strftime("batch_%Y%m%d_%H%M%S"))
    save_button = st.button("Parse, clean, and save", type="primary", use_container_width=True)

if save_button:
    parsed_df, warnings = parse_matches(raw_text, str(match_date), source_batch_id.strip() or "batch_manual")

    if warnings:
        for msg in warnings:
            st.warning(msg)

    if parsed_df.empty:
        st.error("No valid matches were found after parsing and cleaning.")
    else:
        st.subheader("Cleaned matches from this input")
        st.dataframe(parsed_df, use_container_width=True)

        master_df, rejected_existing, accepted_count = append_to_master(parsed_df)

        if accepted_count > 0:
            st.success(f"Saved {accepted_count} new match(es) into the master dataset.")
        else:
            st.info("All cleaned matches from this batch were already present in the master dataset.")

        if not rejected_existing.empty:
            st.info(f"Ignored {len(rejected_existing)} match(es) because they already exist in the saved history.")
            st.dataframe(rejected_existing, use_container_width=True)

        features_df = build_feature_dataset(master_df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Master rows", len(master_df))
        m2.metric("Feature rows", len(features_df))
        m3.metric("Teams seen", len(pd.unique(pd.concat([master_df["home_team"], master_df["away_team"]]))))

        st.subheader("Master dataset preview")
        st.dataframe(master_df.tail(20), use_container_width=True)

        st.subheader("Feature dataset preview")
        st.dataframe(features_df.tail(20), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download master dataset CSV",
                data=master_df.to_csv(index=False).encode("utf-8"),
                file_name="matches_master.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download feature dataset CSV",
                data=features_df.to_csv(index=False).encode("utf-8"),
                file_name="matches_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()

st.subheader("Current saved data")
master_saved = read_master()
if master_saved.empty:
    st.info("No saved master dataset yet. Paste data above and save your first batch.")
else:
    st.dataframe(master_saved.tail(20), use_container_width=True)

    features_saved = build_feature_dataset(master_saved)
    st.write("Saved feature dataset preview")
    st.dataframe(features_saved.tail(20), use_container_width=True)

    with st.expander("Useful notes"):
        st.markdown(
            """
- The app removes empty lines and extra spaces automatically.
- Duplicate matches inside one pasted batch are removed.
- Duplicate matches against the saved master dataset are ignored.
- Features for a match are built only from earlier matches, so the dataset avoids data leakage.
- The target for later modeling is `target_total_goals` and also `target_total_class`.
            """
        )
