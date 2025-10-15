
import streamlit as st
import plotly.express as px
from utils import sidebar_data_loader, get_df, numeric_columns, text_columns

st.set_page_config(page_title="Player Comparison", layout="wide")
sidebar_data_loader()

st.title("⚖️ Player Comparison")

df = get_df()
if df is None:
    st.info("Upload a CSV from the sidebar.")
    st.stop()

name_col = st.selectbox("Player name column", text_columns(df) or df.columns.tolist())
players = sorted(df[name_col].astype(str).unique().tolist())[:5000]

c1, c2, c3 = st.columns(3)
p1 = c1.selectbox("Player A", players)
p2 = c2.selectbox("Player B", players)
metrics = c3.multiselect("Metrics to compare (max 10)", numeric_columns(df), max_selections=10)

if not metrics:
    st.warning("Select at least one metric.")
    st.stop()

A = df[df[name_col].astype(str) == str(p1)][metrics].mean(numeric_only=True)
B = df[df[name_col].astype(str) == str(p2)][metrics].mean(numeric_only=True)

comp = (
    A.to_frame("Player A")
    .join(B.to_frame("Player B"))
    .reset_index()
    .rename(columns={"index": "Metric"})
)

st.subheader("Table")
st.dataframe(comp, use_container_width=True, hide_index=True)

st.subheader("Radar (Player A)")
fig_a = px.line_polar(comp, r="Player A", theta="Metric", line_close=True, template="plotly_white")
st.plotly_chart(fig_a, use_container_width=True)

st.subheader("Radar (Player B)")
fig_b = px.line_polar(comp, r="Player B", theta="Metric", line_close=True, template="plotly_white")
st.plotly_chart(fig_b, use_container_width=True)
