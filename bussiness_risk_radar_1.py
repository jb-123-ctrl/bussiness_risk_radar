import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import urllib.request

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Business Risk Radar", layout="wide")
st.title("📊 Business Risk Radar")
st.write("An interactive dashboard for analyzing business risk datasets.")

# -----------------------------
# Load data from GitHub CSV
# -----------------------------
github_csv_url = "https://raw.githubusercontent.com/jb-123-ctrl/bussiness_risk_radar/main/Fortune%20500%20Companies.csv"

try:
    df = pd.read_csv(github_csv_url)
except Exception:
    st.warning("Cannot load GitHub CSV. Using sample data.")
    df = pd.DataFrame({
        "Risk_Category": ["High", "Medium", "Low", "High", "Low"],
        "Severity": [90, 50, 20, 80, 10],
        "Probability": [0.9, 0.5, 0.2, 0.8, 0.1],
        "Department": ["Finance", "HR", "IT", "Finance", "IT"]
    })

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filter Data")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
filtered_df = df.copy()

# Numeric filters
for col in num_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
    filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

# Categorical filters
for col in cat_cols:
    options = st.sidebar.multiselect(f"{col} values", df[col].unique(), default=list(df[col].unique()))
    filtered_df = filtered_df[filtered_df[col].isin(options)]

# -----------------------------
# Tabs layout
# -----------------------------
tab1, tab2 = st.tabs(["Overview", "Risk Radar"])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("🔍 Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.subheader("ℹ️ Dataset Info")
    buffer = []
    class BufferWriter:
        def write(self, txt):
            buffer.append(txt)
    filtered_df.info(buf=BufferWriter())
    st.text("".join(buffer))

    st.subheader("📏 Summary Statistics")
    st.dataframe(filtered_df.describe(include="all"), use_container_width=True)

    st.subheader("🚨 Missing Values")
    missing = filtered_df.isnull().sum()
    if missing[missing > 0].empty:
        st.write("No missing values found ✅")
    else:
        st.dataframe(missing[missing > 0])

# -----------------------------
# Tab 2: Risk Radar Chart
# -----------------------------
with tab2:
    st.subheader("📡 Risk Radar Chart (Dynamic)")

    if 'Risk_Category' not in filtered_df.columns or len(num_cols) == 0:
        st.warning("Cannot create radar chart: Missing numeric columns or Risk_Category.")
    else:
        # Calculate average numeric values per Risk_Category
        radar_data = filtered_df.groupby('Risk_Category')[num_cols].mean()

        fig_radar = go.Figure()

        for category in radar_data.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data.loc[category].values,
                theta=num_cols,
                fill='toself',
                name=category
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Average Numeric Values per Risk Category"
        )

        st.plotly_chart(fig_radar, use_container_width=True)


