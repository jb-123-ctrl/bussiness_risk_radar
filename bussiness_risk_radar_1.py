import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import urllib.error

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Business Risk Radar", layout="wide")

st.title("📊 Business Risk Radar")
st.write("An interactive dashboard for analyzing risk datasets.")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# -----------------------------
# Load CSV from GitHub or use sample data
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No CSV uploaded. Using sample data.")
    # Replace this with your GitHub raw CSV URL if available
    github_csv_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/yourfile.csv"
    try:
        df = pd.read_csv(github_csv_url)
    except urllib.error.HTTPError:
        st.warning("Cannot load GitHub CSV. Using local sample data.")
        # Sample DataFrame
        df = pd.DataFrame({
            "Risk_Category": ["High", "Medium", "Low", "High", "Low"],
            "Severity": [90, 50, 20, 80, 10],
            "Probability": [0.9, 0.5, 0.2, 0.8, 0.1]
        })

# -----------------------------
# Dataset Overview
# -----------------------------
st.subheader("🔍 Dataset Preview")
st.write(df.head())

# -----------------------------
# Dataset Info
# -----------------------------
st.subheader("ℹ️ Dataset Info")
buffer = []

class BufferWriter:
    def write(self, txt):
        buffer.append(txt)

df.info(buf=BufferWriter())
info_str = "".join(buffer)
st.text(info_str)

# -----------------------------
# Summary Statistics
# -----------------------------
st.subheader("📏 Summary Statistics")
st.write(df.describe(include="all"))

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("🚨 Missing Values")
missing = df.isnull().sum()
st.write(missing[missing > 0])

# -----------------------------
# Correlation Heatmap
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=len(num_cols) <= 10, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Interactive Bar Chart
# -----------------------------
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if cat_cols:
    st.subheader("📊 Bar Chart of Categorical Feature")
    selected_cat = st.selectbox("Select a categorical column:", cat_cols)

    cat_counts = df[selected_cat].value_counts().reset_index()
    cat_counts.columns = [selected_cat, "Count"]
    cat_counts = cat_counts.head(50)  # Limit top 50 categories for performance

    fig2 = px.bar(
        cat_counts,
        x=selected_cat,
        y="Count",
        title=f"Bar Chart of {selected_cat}"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Scatter Plot
# -----------------------------
if len(num_cols) >= 2:
    st.subheader("⚡ Scatter Plot")
    x_axis = st.selectbox("X-axis:", num_cols, index=0)
    y_axis = st.selectbox("Y-axis:", num_cols, index=1)

    # Sample 1000 rows for performance
    df_sample = df.sample(min(1000, len(df)))

    fig3 = px.scatter(df_sample, x=x_axis, y=y_axis, title=f"Scatter: {x_axis} vs {y_axis}")
    st.plotly_chart(fig3, use_container_width=True)
