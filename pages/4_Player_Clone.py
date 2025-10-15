import numpy as np
import pandas as pd
import streamlit as st
from utils import sidebar_data_loader, get_df, numeric_columns, text_columns, cosine_similarity_matrix

st.set_page_config(page_title="Player Clone", layout="wide")
sidebar_data_loader()

st.title("🧬 Player Clone")

df = get_df()
if df is None:
    st.info("Upload a CSV from the sidebar.")
    st.stop()

name_col = st.selectbox("Player name column", text_columns(df) or df.columns.tolist())
metrics = st.multiselect("Metrics for similarity", numeric_columns(df), max_selections=15)

if not metrics:
    st.warning("Pick at least one metric.")
    st.stop()

players = df[name_col].astype(str).tolist()
target = st.selectbox("Target player", sorted(set(players))[:5000])

X = df[metrics].copy().fillna(df[metrics].median(numeric_only=True))
sim = cosine_similarity_matrix(X.to_numpy())

index_by_name = {}
for i, n in enumerate(players):
    index_by_name.setdefault(n, i)

if target not in index_by_name:
    st.error("Target player not found in dataset.")
    st.stop()

ti = index_by_name[target]
scores = sim[ti]

res = pd.DataFrame({"player": players, "similarity": scores})
res = res[res["player"] != target].sort_values("similarity", ascending=False)

k = st.slider("How many similar players?", 5, 30, 10)
st.subheader(f"Most similar to: {target}")
st.dataframe(res.head(k), use_container_width=True, hide_index=True)

