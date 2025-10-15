# pages/1_📊_Stats_Dashboard.py
import streamlit as st
import plotly.express as px
from utils import sidebar_data_loader, get_df, filter_sidebar

st.set_page_config(page_title="Stats Dashboard", page_icon="📊", layout="wide")

# 1️⃣ CSV uploader (persistent)
sidebar_data_loader()

# 2️⃣ Load dataset
df = get_df()
if df is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# 3️⃣ Global filters (shared across all pages)
filtered_df, picks = filter_sidebar(df)

# 4️⃣ Example plot
st.title("📊 Stats Dashboard")
st.write(f"Filtered dataset: {len(filtered_df):,} rows")

metric = st.selectbox("Select metric to visualize", [c for c in filtered_df.columns if filtered_df[c].dtype != 'object'])
group_col = st.selectbox("Group by (optional)", [None] + list(filtered_df.columns))

if metric:
    if group_col:
        fig = px.bar(filtered_df.groupby(group_col)[metric].mean().reset_index(),
                     x=group_col, y=metric, title=f"{metric} by {group_col}")
    else:
        fig = px.histogram(filtered_df, x=metric, nbins=40, title=f"Distribution of {metric}")
    st.plotly_chart(fig, use_container_width=True)
