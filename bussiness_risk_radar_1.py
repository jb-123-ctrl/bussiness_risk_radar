import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Load Dataset
# ----------------------------
st.title("📊 Business Risk Radar Dashboard")

df = pd.read_csv("archive (1).zip", compression="zip")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("### Shape of dataset:", df.shape)
st.write("### Columns:", df.columns)

st.subheader("Dataset Info")
buffer = []
df.info(buf=buffer.append)  # capture info as text
info_str = "\n".join(buffer)
st.text(info_str)

st.subheader("Summary Statistics")
st.write(df.describe())

# ----------------------------
# Exploratory Data Analysis
# ----------------------------
st.header("Exploratory Data Analysis (EDA)")

st.subheader("Distribution of Revenue")
fig = px.histogram(df, x="revenue_mil", nbins=30, title="Distribution of Revenue")
st.plotly_chart(fig)

st.subheader("Distribution of Risk Score")
fig = px.histogram(df, x="risk_score", nbins=30, title="Distribution of Risk Score")
st.plotly_chart(fig)

st.subheader("Revenue vs Risk Score")
fig = px.scatter(df, x="revenue_mil", y="risk_score", color="industry", title="Revenue vs Risk Score")
st.plotly_chart(fig)

# ----------------------------
# Correlation Analysis
# ----------------------------
st.header("Correlation Analysis")

st.subheader("Correlation Heatmap")
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

# ----------------------------
# Predictive Modeling
# ----------------------------
st.header("Predictive Modeling: Revenue vs Risk Score")

X = df[["revenue_mil"]]
y = df["risk_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

st.subheader("Actual vs Predicted Risk Score")
fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted Risk Score")
st.plotly_chart(fig)

# ----------------------------
# Feature Importance (Coefficient)
# ----------------------------
st.header("Feature Importance")
coef_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
st.dataframe(coef_df)

