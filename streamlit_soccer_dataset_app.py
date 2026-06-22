import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Soccer TG Counter System", layout="wide")


DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)

MASTER_PATH = DATA_DIR / "matches_master.csv"
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
    "hit_condition",
    "result",
    "goal_diff",
    "home_team_counter",
    "away_team_counter",
    "match_key",
    "created_at",
]


st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#0f172a 0%,#111827 100%); color: #e5e7eb;}
    .main-card {
        background: rgba(17,24,39,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.45rem;
    }
    .caption-small {
        font-size: 0.84rem;
        color: #cbd5e1;
    }
    div[data-testid="stMetric"] {
        background: rgba(17,24,39,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        padding: 14px;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

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


def get_hit_condition(total_goals: int) -> str:
    if total_goals == 0:
        return "TG0"
    if total_goals == 5:
        return "TG5"
    if total_goals >= 6:
        return "TG6+"
    return "No Hit"


def is_hit(total_goals: int) -> bool:
    return total_goals == 0 or total_goals == 5 or total_goals >= 6


def stable_hash(*parts: str) -> str:
    payload = "|".join(str(p).strip() for p in parts)
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


# ============================================================
# FILE STORAGE
# ============================================================

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
    for path in [MASTER_PATH, STATE_PATH]:
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
            "match_id",
            "cycle_id",
            "week_number",
            "batch_match_number",
            "global_order",
            "home_goals",
            "away_goals",
            "total_goals",
            "goal_diff",
            "home_team_counter",
            "away_team_counter",
        ]

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


# ============================================================
# PARSE INPUT
# ============================================================

def split_input_into_week_sections(
    raw_text: str,
    fallback_week_number: int
) -> List[Tuple[int, List[str]]]:

    raw_lines = [
        re.sub(r"\s+", " ", ln).strip()
        for ln in raw_text.splitlines()
    ]

    sections = []
    current_week = None
    current_lines = []

    for line in raw_lines:
        if not line:
            continue

        detected_week = week_header_value(line)

        if detected_week is not None and line.lower().startswith("english league"):
            if current_lines:
                sections.append(
                    (
                        int(current_week if current_week is not None else fallback_week_number),
                        current_lines,
                    )
                )
                current_lines = []

            current_week = detected_week
            continue

        if is_noise_line(line):
            continue

        if current_week is None:
            current_week = int(fallback_week_number)

        current_lines.append(line)

    if current_lines:
        sections.append(
            (
                int(current_week if current_week is not None else fallback_week_number),
                current_lines,
            )
        )

    return sections


def parse_matches(
    raw_text: str,
    fallback_week_number: int,
    batch_id: str
) -> Tuple[pd.DataFrame, List[str]]:

    warnings = []
    sections = split_input_into_week_sections(raw_text, fallback_week_number)

    if not sections:
        return pd.DataFrame(), ["No usable match lines were found after cleaning."]

    records = []

    if len(sections) > 1:
        warnings.append(f"Detected {len(sections)} week sections.")

    chronological_sections = list(reversed(sections))
    block_no = 0

    for section_week, section_lines in chronological_sections:
        remainder = len(section_lines) % 4

        if remainder:
            warnings.append(
                f"Week {section_week}: ignored last {remainder} line(s). A match needs 4 lines."
            )
            section_lines = section_lines[: len(section_lines) - remainder]

        section_records = []

        for i in range(0, len(section_lines), 4):
            block_no += 1

            home_team_raw, home_goals_raw, away_goals_raw, away_team_raw = section_lines[i:i + 4]

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
                warnings.append(f"Week {section_week}, block {block_no}: home and away team are the same.")
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
                    "hit_condition": get_hit_condition(total_goals),
                    "result": result_code(home_goals, away_goals),
                    "goal_diff": home_goals - away_goals,
                    "created_at": now_iso(),
                }
            )

        if section_records:
            section_df = pd.DataFrame(section_records)
            section_df = section_df.iloc[::-1].reset_index(drop=True)
            records.extend(section_df.to_dict("records"))

    df = pd.DataFrame(records)

    if df.empty:
        return df, warnings

    before = len(df)

    df["_batch_dedupe_key"] = (
        df["week_number"].astype(str)
        + "|"
        + df["home_team"]
        + "|"
        + df["away_team"]
        + "|"
        + df["home_goals"].astype(str)
        + "|"
        + df["away_goals"].astype(str)
    )

    df = df.drop_duplicates(subset=["_batch_dedupe_key"], keep="first").reset_index(drop=True)

    removed = before - len(df)

    if removed:
        warnings.append(f"Removed {removed} duplicate match(es) inside this pasted input.")

    df["batch_match_number"] = np.arange(1, len(df) + 1)

    return df.drop(columns=["_batch_dedupe_key"]), warnings


# ============================================================
# CYCLE LOGIC
# ============================================================

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

    cycles = []

    for _, row in out.iterrows():
        current_week = int(row["week_number"])

        if prev_week is not None and current_week < prev_week:
            current_cycle += 1

        cycles.append(current_cycle)
        prev_week = current_week

    out["cycle_id"] = cycles

    return out


# ============================================================
# EXACT TWO-COLUMN COUNTER LOGIC
# ============================================================

def apply_pair_counters(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return master_df.copy()

    df = master_df.copy().sort_values("global_order").reset_index(drop=True)

    df["home_team_counter"] = 0
    df["away_team_counter"] = 0

    current_cycle = None
    team_counter = {}

    for idx, row in df.iterrows():
        row_cycle = int(row["cycle_id"])

        if current_cycle is None or row_cycle != current_cycle:
            current_cycle = row_cycle
            team_counter = {}

        home_team = row["home_team"]
        away_team = row["away_team"]
        total_goals = int(row["total_goals"])

        if home_team not in team_counter:
            team_counter[home_team] = 0

        if away_team not in team_counter:
            team_counter[away_team] = 0

        if is_hit(total_goals):
            team_counter[home_team] = 0
            team_counter[away_team] = 0
        else:
            team_counter[home_team] += 1
            team_counter[away_team] += 1

        df.at[idx, "home_team_counter"] = team_counter[home_team]
        df.at[idx, "away_team_counter"] = team_counter[away_team]

    return df[MASTER_COLUMNS].copy()


# ============================================================
# SAVE NEW MATCHES
# ============================================================

def append_to_master(new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    master = read_master()

    new_df = assign_cycle_ids(master, new_df)

    new_df["match_key"] = (
        new_df["cycle_id"].astype(int).astype(str)
        + "|"
        + new_df["week_number"].astype(int).astype(str)
        + "|"
        + new_df["home_team"]
        + "|"
        + new_df["away_team"]
        + "|"
        + new_df["home_goals"].astype(int).astype(str)
        + "|"
        + new_df["away_goals"].astype(int).astype(str)
    )

    existing_keys = set(master["match_key"].astype(str)) if not master.empty else set()

    duplicate_mask = new_df["match_key"].astype(str).isin(existing_keys)

    rejected_existing = new_df[duplicate_mask].copy()
    accepted = new_df[~duplicate_mask].copy()

    if accepted.empty:
        return master, rejected_existing, 0

    next_match_id = (
        1
        if master.empty
        else int(pd.to_numeric(master["match_id"], errors="coerce").dropna().max()) + 1
    )

    next_global_order = (
        1
        if master.empty
        else int(pd.to_numeric(master["global_order"], errors="coerce").dropna().max()) + 1
    )

    accepted = accepted.reset_index(drop=True)

    accepted.insert(0, "match_id", range(next_match_id, next_match_id + len(accepted)))
    accepted.insert(5, "global_order", range(next_global_order, next_global_order + len(accepted)))

    for col in MASTER_COLUMNS:
        if col not in accepted.columns:
            accepted[col] = np.nan

    master = pd.concat([master, accepted[MASTER_COLUMNS]], ignore_index=True)
    master = master.sort_values("global_order").reset_index(drop=True)

    master = apply_pair_counters(master)

    save_master(master)

    return master, rejected_existing, len(accepted)


# ============================================================
# METRICS
# ============================================================

def get_metrics() -> dict:
    master = read_master()

    if master.empty:
        return {
            "dataset_rows": 0,
            "teams_seen": 0,
            "cycles_seen": 0,
            "latest_cycle": 0,
            "latest_week": 0,
        }

    latest = master.sort_values("global_order").iloc[-1]

    teams_seen = len(
        pd.unique(
            pd.concat(
                [master["home_team"], master["away_team"]],
                ignore_index=True,
            )
        )
    )

    return {
        "dataset_rows": len(master),
        "teams_seen": teams_seen,
        "cycles_seen": int(pd.to_numeric(master["cycle_id"], errors="coerce").dropna().max()),
        "latest_cycle": int(latest["cycle_id"]),
        "latest_week": int(latest["week_number"]),
    }


# ============================================================
# UI
# ============================================================

st.title("⚽ Soccer TG Pair Counter System")

st.caption(
    "This system stores match records and uses only two counter columns: "
    "home_team_counter and away_team_counter. "
    "If total goals is 0, 5, or 6+, both teams reset to 0. "
    "Otherwise, each team increments according to its own previous count."
)

left, right = st.columns([4, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="main-card">
            <div class="section-title">Recent matches input</div>
            <div class="caption-small">
                Paste recent results. Format should be 4 lines per match:
                home team, home goals, away goals, away team.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_text = st.text_area(
        "Recent results input",
        height=320,
        placeholder="Paste results here...",
        label_visibility="collapsed",
    )

with right:
    st.markdown(
        '<div class="main-card"><div class="section-title">Controls</div>',
        unsafe_allow_html=True,
    )

    detected_week = parse_week_number_from_text(raw_text) if raw_text else None

    fallback_week_number = st.number_input(
        "Fallback week number",
        min_value=1,
        max_value=38,
        value=int(detected_week) if detected_week else 1,
        step=1,
    )

    batch_id = st.text_input("Batch id", value="batch_manual")

    process_results = st.button("Process input", type="primary", use_container_width=True)

    refresh_system = st.button("Refresh system / start new dataset", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


if refresh_system:
    reset_system()
    st.success("System refreshed. Saved records were cleared.")


metrics = get_metrics()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Dataset rows", metrics["dataset_rows"])
m2.metric("Teams seen", metrics["teams_seen"])
m3.metric("Cycles seen", metrics["cycles_seen"])
m4.metric("Latest cycle / week", f'{metrics["latest_cycle"]} / {metrics["latest_week"]}')


# ============================================================
# PROCESS INPUT
# ============================================================

if process_results:
    if not raw_text.strip():
        st.error("Paste some recent results first.")
    else:
        current_hash = stable_hash(
            raw_text,
            str(fallback_week_number),
            batch_id or "batch_manual",
        )

        persisted_state = load_state()
        processed_hashes = set(persisted_state.get("processed_results_hashes", []))

        if st.session_state.get("last_results_hash") == current_hash or current_hash in processed_hashes:
            last = st.session_state.get(
                "last_results_result",
                persisted_state.get("last_results_result", {}),
            )

            st.warning("This exact results batch was already processed. No records were added again.")

            if last:
                st.info(
                    f"Last result: accepted {last.get('accepted', 0)}, "
                    f"existing duplicates {last.get('existing_duplicates', 0)}, "
                    f"warnings {last.get('warnings', 0)}."
                )

        else:
            parsed_df, warnings = parse_matches(
                raw_text,
                int(fallback_week_number),
                batch_id.strip() or "batch_manual",
            )

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

                processed_hashes = list(
                    dict.fromkeys(
                        list(persisted_state.get("processed_results_hashes", []))
                        + [current_hash]
                    )
                )

                persisted_state["processed_results_hashes"] = processed_hashes[-500:]
                persisted_state["last_results_result"] = result_payload

                save_state(persisted_state)

                if accepted_count > 0:
                    st.success(f"Saved {accepted_count} new match(es) into matches_master.csv.")
                else:
                    st.info("All cleaned matches were already saved before.")

                if len(rejected_existing) > 0:
                    st.info(f"Ignored {len(rejected_existing)} existing duplicate match(es).")


# ============================================================
# DASHBOARD
# ============================================================

master_df = read_master()

if not master_df.empty:
    master_df = apply_pair_counters(master_df)
    save_master(master_df)

st.markdown(
    """
    <div class="main-card">
        <div class="section-title">Last 10 match pairs dashboard</div>
        <div class="caption-small">
            Only the latest 10 stored rows are shown here.
            Every row is saved in matches_master.csv.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if master_df.empty:
    st.info("No records yet. Process results first.")
else:
    dashboard_df = master_df.sort_values("global_order").tail(10)

    st.dataframe(
        dashboard_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

master_download = MASTER_PATH.read_bytes() if MASTER_PATH.exists() else b""
requirements_bytes = b"streamlit\npandas\nnumpy\n"

st.markdown(
    """
    <div class="main-card">
        <div class="section-title">Downloads</div>
    </div>
    """,
    unsafe_allow_html=True,
)

d1, d2 = st.columns(2)

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
        "Download requirements.txt",
        data=requirements_bytes,
        file_name="requirements.txt",
        mime="text/plain",
        use_container_width=True,
    )
