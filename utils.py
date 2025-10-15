# utils.py
import io
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_bytes: bytes, encodings=("utf-8", "utf-16", "latin-1")) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.StringIO(file_bytes.decode(enc, errors="strict")), sep=None, engine="python")
        except Exception as e:
            last_err = e
    raise last_err

def sidebar_data_loader(key="footverse_df"):
    st.sidebar.header("📁 Upload your Football CSV")
    up = st.sidebar.file_uploader("Drag & drop or browse", type=["csv"], key="uploader")

    col1, col2 = st.sidebar.columns(2)
    load_btn = col1.button("Load", use_container_width=True)
    clear_btn = col2.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.pop(key, None)
        st.session_state.pop(f"{key}_name", None)

    if load_btn and up is not None:
        with st.spinner("Reading CSV…"):
            df = _read_csv_bytes(up.getvalue())
            st.session_state[key] = df
            st.session_state[f"{key}_name"] = up.name

    if key in st.session_state:
        df = st.session_state[key]
        name = st.session_state.get(f"{key}_name", "uploaded.csv")
        st.sidebar.success(f"Loaded {name} — {len(df):,} rows × {len(df.columns):,} cols")
        with st.sidebar.expander("Preview (top 10)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.sidebar.info("No dataset loaded yet.")

def get_df(key="footverse_df"):
    return st.session_state.get(key)

def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def text_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]

def cosine_similarity_matrix(X: np.ndarray):
    X = X.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norms
    return Xn @ Xn.T
# --- PUT THIS IN utils.py (near the bottom) ---
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand = cand.lower()
        # exact match
        if cand in cols: 
            return cols[cand]
        # loose contains
        for k, v in cols.items():
            if cand in k:
                return v
    return None

def _ensure_age(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # Try to find an Age column; otherwise compute from DOB if available
    age_col = _find_col(df, ["age"])
    if age_col:
        return df, age_col

    dob_col = _find_col(df, ["dob", "date of birth", "birth date", "birthday", "yob"])
    if dob_col:
        def _to_age(v):
            try:
                d = pd.to_datetime(v, errors="coerce")
                if pd.isna(d): 
                    return np.nan
                today = pd.Timestamp.today().normalize()
                return (today - d).days // 365
            except Exception:
                return np.nan
        new = df.copy()
        new["__age__"] = new[dob_col].apply(_to_age)
        return new, "__age__"
    return df, None

def _position_bucket(pos: str) -> str | None:
    if not isinstance(pos, str): 
        return None
    p = pos.upper()
    if "GK" in p or "GOALKEEP" in p:
        return "GK"
    if any(k in p for k in ["CB","RB","LB","RWB","LWB","DEF","DF","BACK","CBR","CBL"]):
        return "DF"
    if any(k in p for k in ["DM","CM","AM","RM","LM","MID","MF","MEZZ","8","6","10"]):
        return "MF"
    if any(k in p for k in ["ST","CF","FW","ATT","LW","RW","WINGER","9"]):
        return "FW"
    return None

def add_position_bucket(df: pd.DataFrame, main_pos_col: str) -> pd.DataFrame:
    new = df.copy()
    new["__bucket__"] = new[main_pos_col].apply(_position_bucket)
    return new

def filter_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Builds the sidebar UI and returns (filtered_df, selections_dict).
    Expects 'Main Position' column to exist (for bucket GK/DF/MF/FW).
    Tries to find League, Team, Nationality, Age, Minutes columns loosely.
    """
    # Column resolution
    league_col = _find_col(df, ["league", "competition", "tournament", "division"])
    team_col   = _find_col(df, ["team", "club", "squad"])
    nat_col    = _find_col(df, ["nationality", "country", "citizenship"])
    pos_col    = _find_col(df, ["Main Position"]) or _find_col(df, ["position", "main position"])
    minutes_col= _find_col(df, ["minutes", "mins", "min played", "minutes played", "time played", "mp"])

    if not pos_col:
        st.sidebar.error("Couldn't find a **Main Position** column. Make sure your CSV has one.")
        pos_col = "Main Position"  # keep UI working; won't filter without real data

    # Derive/resolve Age
    df_age, age_col = _ensure_age(df)

    # Build choices
    st.sidebar.markdown("### 🎯 Refine Your Search")

    if league_col:
        leagues = sorted([x for x in df[league_col].dropna().astype(str).unique() if x != ""])
        sel_leagues = st.sidebar.multiselect("🌍 Select Leagues", leagues, placeholder="Pick your favorite leagues")
    else:
        sel_leagues = []

    if team_col:
        teams = sorted([x for x in df[team_col].dropna().astype(str).unique() if x != ""])
        sel_teams = st.sidebar.multiselect("🏆 Choose Teams", teams, placeholder="Pick your favorite teams")
    else:
        sel_teams = []

    if nat_col:
        nats = sorted([x for x in df[nat_col].dropna().astype(str).unique() if x != ""])
        sel_nats = st.sidebar.multiselect("🧭 Select Nationalities", nats, placeholder="Filter by nationality")
    else:
        sel_nats = []

    # Position buckets
    st.sidebar.markdown("**🧭 Player Positions**")
    # segmented control if available; else buttons:
    options = ["GK","DF","MF","FW"]
    if hasattr(st, "segmented_control"):
        sel_bucket = st.sidebar.segmented_control("Position Group", options, selection_mode="single", label_visibility="collapsed")
        sel_buckets = [sel_bucket] if sel_bucket else []
    else:
        # simple toggle buttons
        cols = st.sidebar.columns(4)
        picks = []
        for opt, c in zip(options, cols):
            if c.button(opt, use_container_width=True):
                picks = [opt]  # single-select behavior
        sel_buckets = picks

    # Age slider
    if age_col:
        age_min = int(np.nanmax([15, np.floor(df_age[age_col].min(skipna=True) if pd.api.types.is_numeric_dtype(df_age[age_col]) else 15)]))
        age_max = int(np.nanmin([50, np.ceil(df_age[age_col].max(skipna=True) if pd.api.types.is_numeric_dtype(df_age[age_col]) else 50)]))
        age_min = max(15, min(age_min, 50))
        age_max = max(age_min, min(age_max, 50))
        rng = st.sidebar.slider("📊 Age Range", 15, 50, (age_min, age_max))
    else:
        rng = (15, 50)

    # Minutes slider
    if minutes_col and pd.api.types.is_numeric_dtype(df[minutes_col]):
        mmax = int(max(0, np.nan_to_num(df[minutes_col], nan=0).max()))
    else:
        mmax = 0
    min_minutes = st.sidebar.slider("⏳ Filter by Minimum Minutes Played", 0, max(100, mmax), min(523, max(0, mmax)) if mmax else 0)

    # Apply filters
    filtered = df_age.copy()

    if league_col and sel_leagues:
        filtered = filtered[filtered[league_col].astype(str).isin(sel_leagues)]

    if team_col and sel_teams:
        filtered = filtered[filtered[team_col].astype(str).isin(sel_teams)]

    if nat_col and sel_nats:
        filtered = filtered[filtered[nat_col].astype(str).isin(sel_nats)]

    # positions from Main Position
    filtered = add_position_bucket(filtered, pos_col)
    if sel_buckets:
        filtered = filtered[filtered["__bucket__"].isin(sel_buckets)]

    # age filter
    if age_col:
        filtered = filtered[(filtered[age_col] >= rng[0]) & (filtered[age_col] <= rng[1])]

    # minutes filter
    if minutes_col and minutes_col in filtered.columns and pd.api.types.is_numeric_dtype(filtered[minutes_col]):
        filtered = filtered[filtered[minutes_col] >= min_minutes]

    selections = {
        "league_col": league_col, "team_col": team_col, "nat_col": nat_col,
        "pos_col": pos_col, "age_col": age_col, "minutes_col": minutes_col,
        "leagues": sel_leagues, "teams": sel_teams, "nationalities": sel_nats,
        "buckets": sel_buckets, "age_range": rng, "min_minutes": min_minutes
    }
    return filtered.reset_index(drop=True), selections

