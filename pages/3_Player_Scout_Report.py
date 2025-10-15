import streamlit as st
import plotly.express as px
from utils import sidebar_data_loader, get_df, numeric_columns, text_columns

st.set_page_config(page_title="Player Scout Report", layout="wide")
sidebar_data_loader()

st.title("🕵️ Player Scout Report")

df = get_df()
if df is None:
    st.info("Upload a CSV from the sidebar.")
    st.stop()

name_col = st.selectbox("Player name column", text_columns(df) or df.columns.tolist())
player = st.selectbox("Select player", sorted(df[name_col].astype(str).unique().tolist())[:5000])

metrics = st.multiselect("Metrics for radar/pizza", numeric_columns(df), max_selections=12)

profile = df[df[name_col].astype(str) == str(player)]
st.subheader(player)
st.write("Basic info (placeholder): team, league, age, position…")
st.dataframe(profile.head(5), use_container_width=True)

if metrics:
    mean_vals = profile[metrics].mean(numeric_only=True).reset_index()
    mean_vals.columns = ["metric", "value"]
    fig = px.line_polar(mean_vals, r="value", theta="metric", line_close=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pick metrics to draw a radar chart.")

st.text_area("Scout Notes", placeholder="Enter qualitative notes about the player…")

