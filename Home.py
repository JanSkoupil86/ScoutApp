# Home.py
import streamlit as st
from utils import sidebar_data_loader, get_df, filter_sidebar

st.set_page_config(page_title="Footverse", page_icon="⚽", layout="wide")

# 1) Keep the CSV uploader in the sidebar
sidebar_data_loader()

st.title("📊 Stats / Screener (demo)")

# 2) Get the uploaded dataframe
df = get_df()
if df is None:
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

# 3) >>> THIS draws the 'Refine Your Search' sidebar <<<
filtered_df, picks = filter_sidebar(df)

# 4) Show results on the main area
st.success(f"{len(filtered_df):,} rows match your filters.")
st.dataframe(filtered_df.head(30), use_container_width=True)
