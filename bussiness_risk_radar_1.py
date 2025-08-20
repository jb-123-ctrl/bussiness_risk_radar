import streamlit as st
import pandas as pd
import numpy as np
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
tab1, tab2 = st.tabs(["Overview", "Analysis"])

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
# Tab 2: Analysis
# -----------------------------
with tab2:
    st.subheader("📌 Key Insights")
    st.markdown(f"- Total Rows: **{len(filtered_df)}**")
    st.markdown(f"- Total Columns: **{len(filtered_df.columns)}**")
    
    if 'Risk_Category' in filtered_df.columns:
        st.markdown(f"- Risk Categories: **{', '.join(filtered_df['Risk_Category'].unique())}**")
    
    if num_cols:
        st.markdown(f"- Numeric Columns: **{', '.join(num_cols)}**")
        # Display correlations
        corr_matrix = filtered_df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corr = upper.stack().sort_values(ascending=False).head(10)
        st.subheader("Top Correlations")
        if not top_corr.empty:
            st.dataframe(top_corr)
        else:
            st.write("No significant correlations found.")

    if cat_cols:
        st.subheader("Categorical Column Distribution")
        for col in cat_cols:
            counts = filtered_df[col].value_counts()
            st.markdown(f"**{col}**")
            st.dataframe(counts)

    st.markdown("---")
    st.info("This dashboard provides an interactive overview and analysis of business risk data. For advanced insights, visual plots can be added in the future.")

