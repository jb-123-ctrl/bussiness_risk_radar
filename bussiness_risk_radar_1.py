import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="Business Risk Radar", layout="wide")
st.title("📊 Business Risk Radar Dashboard")

# ------------------------
# Load dataset
# ------------------------
st.header("1. Load Dataset")
df = pd.read_csv("archive (1).zip", compression="zip")
st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------------
# Dataset Info
# ------------------------
st.header("2. Dataset Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": df.notnull().sum().values,
    "Dtype": df.dtypes.values
})
st.dataframe(info_df)

st.subheader("Shape of Dataset")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ------------------------
# Descriptive Statistics
# ------------------------
st.header("3. Descriptive Statistics")
st.dataframe(df.describe())

# ------------------------
# Visualizations
# ------------------------
st.header("4. Exploratory Data Analysis")

# Example: Histogram of numeric column
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    selected_num = st.selectbox("Select a numeric column for histogram:", num_cols)
    fig = px.histogram(df, x=selected_num, nbins=30, title=f"Histogram of {selected_num}")
    st.plotly_chart(fig, use_container_width=True)

# Example: Bar chart of categorical column
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if cat_cols:
    selected_cat = st.selectbox("Select a categorical column for bar chart:", cat_cols)
    fig2 = px.bar(df[selected_cat].value_counts().reset_index(),
                  x="index", y=selected_cat,
                  title=f"Bar Chart of {selected_cat}")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# Machine Learning Model
# ------------------------
st.header("5. Predictive Model (Linear Regression)")

if len(num_cols) >= 2:
    target = st.selectbox("Select target variable:", num_cols)
    features = st.multiselect("Select feature(s):", [col for col in num_cols if col != target])

    if features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Scatter plot of actual vs predicted
        fig3 = px.scatter(x=y_test, y=y_pred,
                          labels={'x': "Actual", 'y': "Predicted"},
                          title="Actual vs Predicted")
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Not enough numeric columns for regression model.")

