# Home.py
import streamlit as st
from utils import sidebar_data_loader, get_df

st.set_page_config(page_title="Footverse", page_icon="⚽", layout="wide")

try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

sidebar_data_loader()

st.title("🚀 Welcome to Footverse!")
st.markdown("""
Football isn’t just a game—it's **numbers, patterns, and insights**.  
**Footverse** turns raw CSV data into an interactive scouting & analytics workspace.
""")

st.divider()
st.header("🔍 What You Can Do")
st.markdown("""
- 📊 **Stats Dashboard** — Explore distributions and grouped summaries.
- ⚖️ **Player Comparison** — Side-by-side metrics & radar charts.
- 🕵️ **Player Scout Report** — Single player profile with notes.
- 🧬 **Player Clone** — Find similar players via cosine similarity.
- 🛠️ **Help** — Data tips and next steps.
""")

df = get_df()
if df is not None:
    st.subheader("Quick Preview")
    st.caption("First 30 rows · Scroll for more")
    st.dataframe(df.head(30), use_container_width=True)
else:
    st.info("Upload a CSV from the sidebar to begin.")

