import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Business Risk Radar", layout="wide")
st.title("📊 Business Risk Radar")
st.write("An interactive dashboard for analyzing business risk datasets.")

# -----------------------------
# Load sample dataset
# -----------------------------
# Replace this with your GitHub CSV raw URL if needed
github_csv_url = "https://raw.githubusercontent.com/jb-123-ctrl/bussiness_risk_radar/refs/heads/main/Fortune%20500%20Companies.csv"

try:
    df = pd.read_csv(github_csv_url)
except Exception:
    st.warning("Cannot load GitHub CSV. Using sample data.")
    df = pd.DataFrame({
        "Risk_Category": ["High", "Medium", "Low", "High", "Low"],
        "Severity": [90, 50, 20, 80, 10],
        "Probability": [0.9, 0.5, 0.2, 0.8, 0.1],
        "Impact": [95, 60, 30, 85, 15],
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
# Overview / Summary
# -----------------------------
st.subheader("📌 Summary Insights")
st.markdown(f"- Total rows: **{len(filtered_df)}**")
st.markdown(f"- Total columns: **{len(filtered_df.columns)}**")
if 'Risk_Category' in filtered_df.columns:
    risk_counts = filtered_df['Risk_Category'].value_counts()
    st.markdown(f"- Risk Categories: **{', '.join(risk_counts.index)}**")
    st.bar_chart(risk_counts)

if num_cols:
    st.markdown(f"- Numeric Columns: **{', '.join(num_cols)}**")

# -----------------------------
# Radar Chart for Numeric Risk Metrics
# -----------------------------
st.subheader("📡 Radar Chart of Risk Metrics")

if len(filtered_df) > 0 and num_cols:
    categories = num_cols
    values = filtered_df[categories].mean().values.tolist()
    
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Average Risk Metrics',
        line=dict(color='red'),
        marker=dict(size=8)
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(10, max(values)*1.2)])
        ),
        showlegend=True,
        title="Average Risk Metrics (Dynamic with Filters)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("No numeric data available to display radar chart.")

# -----------------------------
# Optional: Display filtered data
# -----------------------------
with st.expander("Show Filtered Data"):
    st.dataframe(filtered_df)
