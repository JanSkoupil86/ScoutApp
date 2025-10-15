import streamlit as st
from utils import sidebar_data_loader

st.set_page_config(page_title="Help", layout="wide")
sidebar_data_loader()

st.title("🛠️ Help")
st.markdown("""
**CSV tips (flexible):**
- Include one identity column for players (e.g., `player`, `name`, `full_name`).
- Include numeric stat columns (e.g., `xG`, `shots`, `tackles`, `progressive_passes`).
- Missing values are okay — similarity/comparison fill with **median** for now.

**Next steps to add:**
- Column aliasing (map your headers to standard names).
- Per-90 rates & percentile ranking.
- Position templates & weight profiles.
- Save/download shortlists and custom reports.
""")

