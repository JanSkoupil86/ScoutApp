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

