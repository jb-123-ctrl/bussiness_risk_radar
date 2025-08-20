import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Business Risk Radar", layout="wide")
st.title("📊 Business Risk Radar")
st.write("An interactive dashboard for analyzing business risk datasets.")

# -----------------------------
# Load CSV directly from GitHub
# -----------------------------
github_csv_url = "https://raw.githubusercontent.com/jb-123-ctrl/bussiness_risk_radar/refs/heads/main/Fortune%20500%20Companies.csv"

try:
    df = pd.read_csv(github_csv_url)
except Exception as e:
    st.error(f"Failed to load GitHub CSV: {e}")
    # Fallback to sample data
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
tab1, tab2, tab3 = st.tabs(["Overview", "Plots", "Analysis"])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("🔍 Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader("ℹ️ Dataset Info")
    buffer = []
    class BufferWriter:
        def write(self, txt):
            buffer.append(txt)
    filtered_df.info(buf=BufferWriter())
    st.text("".join(buffer))

    st.subheader("📏 Summary Statistics")
    st.dataframe(filtered_df.describe(include="all"))

    st.subheader("🚨 Missing Values")
    missing = filtered_df.isnull().sum()
    st.write(missing[missing > 0])

    st.subheader("📌 Key Insights")
    st.markdown(f"- Total Rows: **{len(filtered_df)}**")
    st.markdown(f"- Total Columns: **{len(filtered_df.columns)}**")
    if 'Risk_Category' in filtered_df.columns:
        st.markdown(f"- Risk Categories: **{', '.join(filtered_df['Risk_Category'].unique())}**")
    if num_cols:
        st.markdown(f"- Numeric Columns: **{', '.join(num_cols)}**")

# -----------------------------
# Tab 2: Plots
# -----------------------------
with tab2:
    # Correlation Heatmap
    if num_cols:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(filtered_df[num_cols].corr(), annot=len(num_cols)<=10, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Scatter Plot
        st.subheader("⚡ Scatter Plot")
        x_axis = st.selectbox("X-axis", num_cols, index=0)
        y_axis = st.selectbox("Y-axis", num_cols, index=1)
        df_sample = filtered_df.sample(min(1000, len(filtered_df)))
        fig3 = px.scatter(df_sample, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig3, use_container_width=True)

        # Histogram
        st.subheader("📊 Histogram")
        selected_num = st.selectbox("Select numeric column for histogram", num_cols)
        fig_hist = px.histogram(filtered_df, x=selected_num, nbins=20, title=f"Histogram of {selected_num}")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box Plot
        st.subheader("📦 Box Plot")
        selected_num_box = st.selectbox("Select numeric column for box plot", num_cols, index=0)
        fig_box = px.box(filtered_df, y=selected_num_box, points="all", title=f"Box Plot of {selected_num_box}")
        st.plotly_chart(fig_box, use_container_width=True)

        # Radar Chart
        st.subheader("📡 Radar Chart")
        categories = num_cols
        values = filtered_df[categories].mean().values.tolist()
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Average Values'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Radar Chart of Average Numeric Columns"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Categorical Plots
    if cat_cols:
        st.subheader("📊 Bar Chart")
        selected_cat = st.selectbox("Select categorical column for bar chart", cat_cols)
        cat_counts = filtered_df[selected_cat].value_counts().reset_index()
        cat_counts.columns = [selected_cat, "Count"]
        fig_bar = px.bar(cat_counts, x=selected_cat, y="Count", title=f"Bar Chart of {selected_cat}")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🥧 Pie Chart")
        selected_cat_pie = st.selectbox("Select categorical column for pie chart", cat_cols, index=0)
        pie_counts = filtered_df[selected_cat_pie].value_counts().reset_index()
        pie_counts.columns = [selected_cat_pie, "Count"]
        fig_pie = px.pie(pie_counts, names=selected_cat_pie, values="Count", title=f"Pie Chart of {selected_cat_pie}")
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Tab 3: Analysis
# -----------------------------
with tab3:
    st.subheader("📌 Insights & Analysis")
    st.write("- Total rows:", len(filtered_df))
    st.write("- Total columns:", len(filtered_df.columns))

    if 'Risk_Category' in filtered_df.columns:
        risk_counts = filtered_df['Risk_Category'].value_counts()
        st.write("Risk category distribution:")
        st.bar_chart(risk_counts)

    if num_cols:
        st.write("Top correlations:")
        corr_matrix = filtered_df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corr = upper.stack().sort_values(ascending=False).head(10)
        st.write(top_corr)
