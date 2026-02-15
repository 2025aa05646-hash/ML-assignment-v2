import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("ML Assignment 2 - Classification Models Dashboard")

st.write("Upload dataset and select model to evaluate")

# Model selection dropdown
model_name = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load model function
def load_model(name):
    model_path = f"models/{name}.pkl"
    return joblib.load(model_path)

# When file uploaded
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Assume last column is target
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # --------------------------
    # FIX: Consistent encoding using factorize
    # --------------------------
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]

    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # --------------------------
    # Load selected model
    # --------------------------
    model = load_model(model_name)

    # Predictions
    y_pred = model.predict(X)

    # AUC Score
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0.0

    # Metrics calculation
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)

    # --------------------------
    # Display metrics
    # --------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("Precision", f"{precision:.4f}")

    col2.metric("Recall", f"{recall:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")

    col3.metric("AUC Score", f"{auc:.4f}")
    col3.metric("MCC Score", f"{mcc:.4f}")

    # --------------------------
    # Confusion Matrix
    # --------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------
    # Classification Report
    # --------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred)

    st.text(report)

else:
    st.info("Please upload a dataset to continue")
