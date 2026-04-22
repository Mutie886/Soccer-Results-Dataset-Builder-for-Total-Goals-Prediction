import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soccer TG Cycle Dataset", layout="wide")

DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
MASTER_PATH = DATA_DIR / "matches_master.csv"
STATE_PATH = DATA_DIR / "system_state.json"
TG_SUMMARY_PATH = DATA_DIR / "tg_summary_history.csv"

TOTAL_GOAL_VALUES = list(range(7))  # 0..6 where 6 means 6+
MASTER_COLUMNS = [
    "match_id", "cycle_id", "week_number", "batch_id", "batch_match_number", "global_order",
    "home_team", "home_goals", "away_goals", "away_team", "total_goals", "total_goals_bucket",
    "result", "goal_diff", "match_key", "created_at",
]
for side in ["home_team", "away_team"]:
    for g in TOTAL_GOAL_VALUES:
        MASTER_COLUMNS.append(f"{side}_tg_{g}_counter")

TG_SUMMARY_COLUMNS = [
    "cycle_id", "week_number", "team",
    *[f"current_tg_{g}" for g in TOTAL_GOAL_VALUES],
    *[f"max_tg_{g}" for g in TOTAL_GOAL_VALUES],
    *[f"avg_gap_tg_{g}" for g in TOTAL_GOAL_VALUES],
    *[f"freq_tg_{g}" for g in TOTAL_GOAL_VALUES],
    "most_frequent_tg", "highest_frequency"
]

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#0f172a 0%,#111827 100%); color: #e5e7eb;}
    .main-card {background: rgba(17,24,39,0.88); border: 1px solid rgba(148,163,184,0.18); border-radius: 18px; padding: 18px; margin-bottom: 14px; box-shadow: 0 14px 30px rgba(0,0,0,0.24);}
    .section-title {font-size: 1.06rem; font-weight: 700; margin-bottom: 0.45rem; color: #f8fafc;}
    .caption-small {font-size: 0.83rem; color: #cbd5e1;}
    div[data-testid="stMetric"] {background: rgba(17,24,39,0.88); border: 1px solid rgba(148,163,184,0.18); padding: 14px 14px 10px 14px; border-radius: 16px;}
    .small-help {font-size:0.80rem; color:#cbd5e1; line-height:1.35;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def total_goal_bucket(total_goals: int) -> int:
    return min(int(total_goals), 6)


def stable_hash(*parts: str) -> str:
    payload = "|".join(part.strip() for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_week_number_from_text(raw_text: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", raw_text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def week_header_value(line: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", str(line), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def is_noise_line(line: str) -> bool:
    low = str(line).lower().strip()
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
    for path in [MASTER_PATH, TG_SUMMARY_PATH, STATE_PATH]:
        if path.exists():
            path.unlink()
    for key in ["last_results_hash", "last_results_result"]:
        st.session_state.pop(key, None)
    save_state({})


def read_master() -> pd.DataFrame:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH)
        for col in MASTER_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[MASTER_COLUMNS]
        numeric_cols = [
            "match_id", "cycle_id", "week_number", "batch_match_number", "global_order",
            "home_goals", "away_goals", "total_goals", "total_goals_bucket", "goal_diff",
        ] + [c for c in MASTER_COLUMNS if c.endswith("_counter")]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame(columns=MASTER_COLUMNS)


def save_master(df: pd.DataFrame) -> None:
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[MASTER_COLUMNS].copy()
    df.to_csv(MASTER_PATH, index=False)

def save_tg_summary(df: pd.DataFrame) -> None:
    for col in TG_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[TG_SUMMARY_COLUMNS].copy()
    df.to_csv(TG_SUMMARY_PATH, index=False)


def read_tg_summary() -> pd.DataFrame:
    if TG_SUMMARY_PATH.exists():
        df = pd.read_csv(TG_SUMMARY_PATH)
        for col in TG_SUMMARY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[TG_SUMMARY_COLUMNS]
    return pd.DataFrame(columns=TG_SUMMARY_COLUMNS)


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


def parse_matches(raw_text: str, fallback_week_number: int, batch_id: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    sections = split_input_into_week_sections(raw_text, int(fallback_week_number))
    if not sections:
        return pd.DataFrame(), ["No usable match lines were found after cleaning."]

    records: List[dict] = []
    if len(sections) > 1:
        warnings.append(f"Detected {len(sections)} week sections and assigned week numbers per section.")

    chronological_sections = list(reversed(sections))
    block_no = 0
    for section_week, section_lines in chronological_sections:
        remainder = len(section_lines) % 4
        if remainder:
            warnings.append(f"Week {section_week}: ignored the last {remainder} line(s) because a valid match needs 4 lines.")
            section_lines = section_lines[: len(section_lines) - remainder]

        section_records: List[dict] = []
        for i in range(0, len(section_lines), 4):
            block_no += 1
            home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = section_lines[i:i+4]
            home_team = normalize_team_name(home_team_raw)
            away_team = normalize_team_name(away_team_raw)
            try:
                home_goals = int(home_goals_raw)
                away_goals = int(away_goals_raw)
            except ValueError:
                warnings.append(f"Week {section_week}, block {block_no}: scores must be integers.")
                continue
            if home_goals < 0 or away_goals < 0:
                warnings.append(f"Week {section_week}, block {block_no}: negative goals are not allowed.")
                continue
            if home_team == away_team:
                warnings.append(f"Week {section_week}, block {block_no}: home team and away team are identical.")
                continue
            total_goals = home_goals + away_goals
            section_records.append(
                {
                    "batch_id": batch_id,
                    "week_number": int(section_week),
                    "home_team": home_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "away_team": away_team,
                    "total_goals": total_goals,
                    "total_goals_bucket": total_goal_bucket(total_goals),
                    "result": result_code(home_goals, away_goals),
                    "goal_diff": home_goals - away_goals,
                    "created_at": now_iso(),
                }
            )
        if not section_records:
            continue
        section_df = pd.DataFrame(section_records)
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
        master_sorted = master.sort_values("global_order")
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


def apply_total_goal_cycle_counters(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        out = master_df.copy()
        for side in ["home_team", "away_team"]:
            for g in TOTAL_GOAL_VALUES:
                out[f"{side}_tg_{g}_counter"] = np.nan
        return out

    df = master_df.copy().sort_values("global_order").reset_index(drop=True)
    for side in ["home_team", "away_team"]:
        for g in TOTAL_GOAL_VALUES:
            df[f"{side}_tg_{g}_counter"] = np.nan

    current_cycle = None
    team_counters: Dict[str, Dict[int, int]] = {}

    def ensure_team(team: str):
        if team not in team_counters:
            team_counters[team] = {g: 0 for g in TOTAL_GOAL_VALUES}

    for idx, row in df.iterrows():
        row_cycle = int(row["cycle_id"])
        if current_cycle is None or row_cycle != current_cycle:
            current_cycle = row_cycle
            team_counters = {}

        bucket = int(row["total_goals_bucket"])
        for side, team_col in [("home_team", "home_team"), ("away_team", "away_team")]:
            team = row[team_col]
            ensure_team(team)
            for g in TOTAL_GOAL_VALUES:
                if g == bucket:
                    team_counters[team][g] = 1
                else:
                    team_counters[team][g] += 1
                df.at[idx, f"{side}_tg_{g}_counter"] = team_counters[team][g]
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[MASTER_COLUMNS].copy()


def append_to_master(new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
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
        return master, rejected_existing, 0

    next_match_id = 1 if master.empty else int(pd.to_numeric(master["match_id"], errors="coerce").dropna().max()) + 1
    next_global_order = 1 if master.empty else int(pd.to_numeric(master["global_order"], errors="coerce").dropna().max()) + 1
    accepted = accepted.reset_index(drop=True)
    accepted.insert(0, "match_id", range(next_match_id, next_match_id + len(accepted)))
    accepted.insert(5, "global_order", range(next_global_order, next_global_order + len(accepted)))

    for col in MASTER_COLUMNS:
        if col not in accepted.columns:
            accepted[col] = np.nan
    master = pd.concat([master, accepted[MASTER_COLUMNS]], ignore_index=True)
    numeric_cols = [
        "match_id", "cycle_id", "week_number", "batch_match_number", "global_order",
        "home_goals", "away_goals", "total_goals", "total_goals_bucket", "goal_diff"
    ]
    for c in numeric_cols:
        master[c] = pd.to_numeric(master[c], errors="coerce")
    master = master.sort_values("global_order").reset_index(drop=True)
    master = apply_total_goal_cycle_counters(master)
    save_master(master)
    save_tg_summary(build_team_tg_summary_history(master))
    return master, rejected_existing, len(accepted)


def build_team_current_dashboard(master_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["team"] + [f"tg_{g}_counter" for g in TOTAL_GOAL_VALUES]
    if master_df.empty:
        return pd.DataFrame(columns=cols)

    df = master_df.sort_values("global_order")
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    rows = []
    for team in sorted(teams):
        team_home = df[df["home_team"] == team].tail(1)
        team_away = df[df["away_team"] == team].tail(1)
        latest_home_go = team_home["global_order"].iloc[0] if not team_home.empty else -1
        latest_away_go = team_away["global_order"].iloc[0] if not team_away.empty else -1
        if latest_home_go == -1 and latest_away_go == -1:
            continue
        latest_row = team_home.iloc[0] if latest_home_go > latest_away_go else team_away.iloc[0]
        side = "home_team" if latest_home_go > latest_away_go else "away_team"
        row = {"team": team}
        for g in TOTAL_GOAL_VALUES:
            row[f"tg_{g}_counter"] = int(pd.to_numeric(latest_row.get(f"{side}_tg_{g}_counter", 0), errors="coerce") or 0)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def build_team_tg_summary(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_cols = ["team"] + [f"TG {g}" for g in TOTAL_GOAL_VALUES]
    max_cols = ["team"] + [f"TG {g}" for g in TOTAL_GOAL_VALUES]
    avg_cols = ["team"] + [f"TG {g}" for g in TOTAL_GOAL_VALUES]
    freq_cols = ["team"] + [f"TG {g}" for g in TOTAL_GOAL_VALUES] + ["Most frequent TG", "Highest frequency"]
    if master_df.empty:
        return (pd.DataFrame(columns=current_cols), pd.DataFrame(columns=max_cols),
                pd.DataFrame(columns=avg_cols), pd.DataFrame(columns=freq_cols))

    df = master_df.copy().sort_values(["global_order"]).reset_index(drop=True)
    teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)))

    current_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    max_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    freq_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    gap_lists: Dict[str, Dict[int, List[int]]] = {t: {g: [] for g in TOTAL_GOAL_VALUES} for t in teams}
    current_cycle = None

    def reset_cycle_maps():
        for t in teams:
            current_map[t] = {g: 0 for g in TOTAL_GOAL_VALUES}

    for _, row in df.iterrows():
        row_cycle = int(row["cycle_id"])
        if current_cycle is None or row_cycle != current_cycle:
            current_cycle = row_cycle
            reset_cycle_maps()

        bucket = int(row["total_goals_bucket"])
        for team in [row["home_team"], row["away_team"]]:
            for g in TOTAL_GOAL_VALUES:
                if g == bucket:
                    prev_val = current_map[team][g]
                    if prev_val > 0:
                        gap_lists[team][g].append(prev_val)
                    current_map[team][g] = 1
                    freq_map[team][g] += 1
                else:
                    current_map[team][g] += 1
                if current_map[team][g] > max_map[team][g]:
                    max_map[team][g] = current_map[team][g]

    current_rows, max_rows, avg_rows, freq_rows = [], [], [], []
    for team in teams:
        current_row = {"team": team}
        max_row = {"team": team}
        avg_row = {"team": team}
        freq_row = {"team": team}
        for g in TOTAL_GOAL_VALUES:
            current_row[f"TG {g}"] = current_map[team][g]
            max_row[f"TG {g}"] = max_map[team][g]
            avg_row[f"TG {g}"] = round(float(np.mean(gap_lists[team][g])), 2) if gap_lists[team][g] else 0.0
            freq_row[f"TG {g}"] = freq_map[team][g]
        most_freq_bucket = min(TOTAL_GOAL_VALUES, key=lambda g: (-freq_map[team][g], g))
        freq_row["Most frequent TG"] = most_freq_bucket
        freq_row["Highest frequency"] = freq_map[team][most_freq_bucket]
        current_rows.append(current_row)
        max_rows.append(max_row)
        avg_rows.append(avg_row)
        freq_rows.append(freq_row)

    return (
        pd.DataFrame(current_rows, columns=current_cols),
        pd.DataFrame(max_rows, columns=max_cols),
        pd.DataFrame(avg_rows, columns=avg_cols),
        pd.DataFrame(freq_rows, columns=freq_cols),
    )



def build_team_tg_summary_history(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame(columns=TG_SUMMARY_COLUMNS)

    df = master_df.copy().sort_values(["global_order"]).reset_index(drop=True)
    teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)))
    current_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    max_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    freq_map: Dict[str, Dict[int, int]] = {t: {g: 0 for g in TOTAL_GOAL_VALUES} for t in teams}
    gap_lists: Dict[str, Dict[int, List[int]]] = {t: {g: [] for g in TOTAL_GOAL_VALUES} for t in teams}
    current_cycle = None
    summary_rows = []

    def reset_cycle_maps():
        for t in teams:
            current_map[t] = {g: 0 for g in TOTAL_GOAL_VALUES}
            max_map[t] = {g: 0 for g in TOTAL_GOAL_VALUES}
            freq_map[t] = {g: 0 for g in TOTAL_GOAL_VALUES}
            gap_lists[t] = {g: [] for g in TOTAL_GOAL_VALUES}

    ordered_weeks = df[["cycle_id", "week_number"]].drop_duplicates().sort_values(["cycle_id", "week_number"]).itertuples(index=False)
    for wk in ordered_weeks:
        cycle_id, week_number = int(wk.cycle_id), int(wk.week_number)
        if current_cycle is None or cycle_id != current_cycle:
            current_cycle = cycle_id
            reset_cycle_maps()
        week_matches = df[(df["cycle_id"] == cycle_id) & (df["week_number"] == week_number)].sort_values("global_order")
        for _, row in week_matches.iterrows():
            bucket = int(row["total_goals_bucket"])
            for team in [row["home_team"], row["away_team"]]:
                for g in TOTAL_GOAL_VALUES:
                    if g == bucket:
                        prev_val = current_map[team][g]
                        if prev_val > 0:
                            gap_lists[team][g].append(prev_val)
                        current_map[team][g] = 1
                        freq_map[team][g] += 1
                    else:
                        current_map[team][g] += 1
                    if current_map[team][g] > max_map[team][g]:
                        max_map[team][g] = current_map[team][g]
        for team in teams:
            row = {"cycle_id": cycle_id, "week_number": week_number, "team": team}
            for g in TOTAL_GOAL_VALUES:
                row[f"current_tg_{g}"] = current_map[team][g]
                row[f"max_tg_{g}"] = max_map[team][g]
                row[f"avg_gap_tg_{g}"] = round(float(np.mean(gap_lists[team][g])), 2) if gap_lists[team][g] else 0.0
                row[f"freq_tg_{g}"] = freq_map[team][g]
            most_freq_bucket = min(TOTAL_GOAL_VALUES, key=lambda g: (-freq_map[team][g], g))
            row["most_frequent_tg"] = most_freq_bucket
            row["highest_frequency"] = freq_map[team][most_freq_bucket]
            summary_rows.append(row)
    return pd.DataFrame(summary_rows, columns=TG_SUMMARY_COLUMNS)

def get_notification_metrics() -> dict:
    master = read_master()
    latest_cycle = 0
    latest_week = 0
    if not master.empty:
        last = master.sort_values("global_order").iloc[-1]
        latest_cycle = int(last["cycle_id"])
        latest_week = int(last["week_number"])
    return {
        "dataset_rows": int(len(master)),
        "teams_seen": int(len(pd.unique(pd.concat([master["home_team"], master["away_team"]], ignore_index=True)))) if not master.empty else 0,
        "cycles_seen": int(pd.to_numeric(master["cycle_id"], errors="coerce").dropna().max()) if not master.empty and pd.to_numeric(master["cycle_id"], errors="coerce").dropna().size else 0,
        "latest_cycle": latest_cycle,
        "latest_week": latest_week,
    }


st.title("⚽ Soccer TG Cycle Dataset Builder")
st.caption("Process match inputs into the main dataset and monitor team-by-team TG 0 to TG 6+ cycle behaviour.")

left, right = st.columns([4, 1], gap="large")
with left:
    st.markdown('<div class="main-card"><div class="section-title">Recent matches input</div><div class="small-help">Paste recent results here. The system cleans the text, stores matches in chronological order, and enriches matches_master.csv with the TG0 to TG6 current counters for both teams in every row.</div>', unsafe_allow_html=True)
    raw_text = st.text_area("Recent results input", height=320, placeholder="Paste the recent results here...", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card"><div class="section-title">Controls</div>', unsafe_allow_html=True)
    detected_week = parse_week_number_from_text(raw_text) if raw_text else None
    fallback_week_number = st.number_input("Fallback week number", min_value=1, max_value=38, value=int(detected_week) if detected_week else 1, step=1)
    batch_id = st.text_input("Batch id", value="batch_manual")
    process_results = st.button("Process input", type="primary", use_container_width=True)
    refresh_system = st.button("Refresh system / start new dataset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if refresh_system:
    reset_system()
    st.success("System refreshed. Saved records were cleared.")

st.markdown('<div class="main-card"><div class="section-title">Notifications dashboard</div></div>', unsafe_allow_html=True)
metrics = get_notification_metrics()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset rows", metrics["dataset_rows"])
m2.metric("Teams seen", metrics["teams_seen"])
m3.metric("Cycles seen", metrics["cycles_seen"])
m4.metric("Latest cycle / week", f"{metrics['latest_cycle']} / {metrics['latest_week']}")

if process_results:
    if not raw_text.strip():
        st.error("Paste some recent results first.")
    else:
        current_hash = stable_hash(raw_text, str(fallback_week_number), batch_id or "batch_manual")
        persisted_state = load_state()
        processed_hashes = set(persisted_state.get("processed_results_hashes", []))
        if st.session_state.get("last_results_hash") == current_hash or current_hash in processed_hashes:
            last = st.session_state.get("last_results_result", persisted_state.get("last_results_result", {}))
            st.warning("This exact results batch was already processed. No records were added again.")
            if last:
                st.info(f"Last result: accepted {last.get('accepted', 0)}, existing duplicates {last.get('existing_duplicates', 0)}, warnings {last.get('warnings', 0)}.")
        else:
            parsed_df, warnings = parse_matches(raw_text, int(fallback_week_number), batch_id.strip() or "batch_manual")
            for msg in warnings:
                st.warning(msg)
            if parsed_df.empty:
                st.error("No valid matches were found after cleaning.")
            else:
                master_df, rejected_existing, accepted_count = append_to_master(parsed_df)
                result_payload = {
                    "accepted": int(accepted_count),
                    "existing_duplicates": int(len(rejected_existing)),
                    "warnings": int(len(warnings)),
                }
                st.session_state["last_results_hash"] = current_hash
                st.session_state["last_results_result"] = result_payload
                persisted_state = load_state()
                processed_hashes = list(dict.fromkeys(list(persisted_state.get("processed_results_hashes", [])) + [current_hash]))
                persisted_state["processed_results_hashes"] = processed_hashes[-500:]
                persisted_state["last_results_result"] = result_payload
                save_state(persisted_state)
                if accepted_count > 0:
                    st.success(f"Saved {accepted_count} new match(es) and rebuilt matches_master.csv.")
                else:
                    st.info("All cleaned matches from this batch were already present in the saved history.")
                if len(rejected_existing) > 0:
                    st.info(f"Ignored {len(rejected_existing)} match(es) already present in saved history.")

master_df = read_master()
summary_history_df = read_tg_summary()
current_df, max_df, avg_df, freq_df = build_team_tg_summary(master_df)

st.markdown('<div class="main-card"><div class="section-title">Current TG pass dashboard</div><div class="caption-small">Each TG column shows the latest running counter for that team. A hit resets that TG to 1; every miss increments it.</div></div>', unsafe_allow_html=True)
if current_df.empty:
    st.info("No team counters yet. Process results to build the dashboard.")
else:
    st.dataframe(current_df, use_container_width=True, hide_index=True)

st.markdown('<div class="main-card"><div class="section-title">Maximum pass dashboard</div><div class="caption-small">For each team and each TG bucket, this shows the highest pass count ever reached from the saved history.</div></div>', unsafe_allow_html=True)
if max_df.empty:
    st.info("No maximum-pass summary available yet.")
else:
    st.dataframe(max_df, use_container_width=True, hide_index=True)

st.markdown('<div class="main-card"><div class="section-title">Average repeat dashboard</div><div class="caption-small">For each TG bucket, this shows the average pass count a team takes before that TG appears again.</div></div>', unsafe_allow_html=True)
if avg_df.empty:
    st.info("No average-repeat summary available yet.")
else:
    st.dataframe(avg_df, use_container_width=True, hide_index=True)

st.markdown('<div class="main-card"><div class="section-title">TG frequency dashboard</div><div class="caption-small">This shows how many times each TG bucket has occurred for each team, plus the minimum TG bucket with the highest frequency.</div></div>', unsafe_allow_html=True)
if freq_df.empty:
    st.info("No TG frequency summary available yet.")
else:
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

st.markdown('<div class="main-card"><div class="section-title">Continuous weekly TG summary records</div><div class="caption-small">This keeps a week-by-week record for every team of the current TG counters, maximum passes, average repeat gaps, and TG frequencies.</div></div>', unsafe_allow_html=True)
if summary_history_df.empty:
    st.info("No continuous weekly summary records yet.")
else:
    st.dataframe(summary_history_df, use_container_width=True, hide_index=True)

master_download = MASTER_PATH.read_bytes() if MASTER_PATH.exists() else b""
summary_download = TG_SUMMARY_PATH.read_bytes() if TG_SUMMARY_PATH.exists() else b""
reqs_bytes = b"streamlit\npandas\nnumpy\n"

st.markdown('<div class="main-card"><div class="section-title">Downloads</div><div class="caption-small">matches_master.csv includes only the core match columns plus the TG0 to TG6 current counters for home and away teams.</div></div>', unsafe_allow_html=True)
d1, d2 = st.columns(2)
with d1:
    st.download_button("Download matches_master.csv", data=master_download, file_name="matches_master.csv", mime="text/csv", use_container_width=True, disabled=not bool(master_download))
with d2:
    st.download_button("Download requirements.txt", data=reqs_bytes, file_name="requirements.txt", mime="text/plain", use_container_width=True)


st.markdown('<div class="main-card"><div class="section-title">Downloads</div><div class="caption-small">Download the enriched match dataset and the continuous weekly TG summary history.</div></div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download matches_master.csv", data=master_download, file_name="matches_master.csv", mime="text/csv", use_container_width=True, disabled=not bool(master_download))
with d2:
    st.download_button("Download tg_summary_history.csv", data=summary_download, file_name="tg_summary_history.csv", mime="text/csv", use_container_width=True, disabled=not bool(summary_download))
with d3:
    st.download_button("Download requirements.txt", data=reqs_bytes, file_name="requirements.txt", mime="text/plain", use_container_width=True)
