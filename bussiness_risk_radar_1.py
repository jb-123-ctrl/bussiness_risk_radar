import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # Dataset Overview
    # -----------------------------
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    st.subheader("ℹ️ Dataset Info")
    buffer = []
    df.info(buf=buffer := type("", (), {"write": buffer.append})())
    info_str = "\n".join(buffer)
    st.text(info_str)

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
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
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
        fig3 = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter: {x_axis} vs {y_axis}")
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin.")


