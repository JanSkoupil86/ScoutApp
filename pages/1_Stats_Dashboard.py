import streamlit as st
import plotly.express as px
from utils import sidebar_data_loader, get_df, numeric_columns, text_columns

st.set_page_config(page_title="Stats Dashboard", layout="wide")
sidebar_data_loader()

st.title("📊 Stats Dashboard")

df = get_df()
if df is None:
    st.info("Upload a CSV from the sidebar to explore stats.")
    st.stop()

num_cols = numeric_columns(df)
txt_cols = text_columns(df)

right, left = st.columns([1, 2])
with right:
    group = st.selectbox("Group by (optional)", ["— none —"] + txt_cols)
    metric = st.selectbox("Metric", num_cols or df.columns.tolist())
    chart = st.radio("Chart", ["Bar", "Histogram", "Box"], horizontal=True)

with left:
    if group != "— none —":
        agg = df.groupby(group, dropna=False)[metric].mean(numeric_only=True).reset_index()
        if chart == "Bar":
            fig = px.bar(agg.sort_values(metric, ascending=False), x=group, y=metric)
        elif chart == "Histogram":
            fig = px.histogram(df, x=metric, color=group, nbins=30, opacity=0.6, barmode="overlay")
        else:
            fig = px.box(df, x=group, y=metric)
    else:
        if chart == "Bar":
            label_col = txt_cols[0] if txt_cols else None
            tmp = df.nlargest(20, metric, keep="all") if metric in df.columns else df.head(20)
            fig = px.bar(tmp, x=metric if metric in df.columns else tmp.columns[0],
                         y=label_col if label_col else tmp.index, orientation="h")
        elif chart == "Histogram":
            fig = px.histogram(df, x=metric if metric in df.columns else df.columns[0], nbins=30)
        else:
            fig = px.box(df, y=metric if metric in df.columns else df.columns[0])

    st.plotly_chart(fig, use_container_width=True)

