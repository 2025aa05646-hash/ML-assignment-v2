import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load scaler and encoders
# =========================

scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# =========================
# Page title
# =========================

st.title("ML Assignment 2 - Classification Dashboard")

# =========================
# Model selection
# =========================

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

# =========================
# Load model
# =========================

def load_model(name):
    return joblib.load(f"models/{name}.pkl")

# =========================
# File uploader
# =========================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, header=None)

    df.columns = [
        'age','workclass','fnlwgt','education','education-num',
        'marital-status','occupation','relationship','race',
        'sex','capital-gain','capital-loss','hours-per-week',
        'native-country','income'
    ]

    st.write("Dataset Preview")
    st.write(df.head())

    X = df.drop("income", axis=1)
    y = df["income"]

    # =========================
    # Apply LabelEncoders
    # =========================

    for col in X.columns:
        if col in label_encoders:
            le = label_encoders[col]
            X[col] = le.transform(X[col].astype(str))

    if "income" in label_encoders:
        y = label_encoders["income"].transform(y.astype(str))

    # =========================
    # Apply Scaling
    # =========================

    X_scaled = scaler.transform(X)

    # =========================
    # Load model and predict
    # =========================

    model = load_model(model_name)

    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:,1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0.0

    # =========================
    # Metrics
    # =========================

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", round(accuracy,4))
    st.write("Precision:", round(precision,4))
    st.write("Recall:", round(recall,4))
    st.write("F1 Score:", round(f1,4))
    st.write("AUC Score:", round(auc,4))
    st.write("MCC Score:", round(mcc,4))

    # =========================
    # Confusion Matrix
    # =========================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    st.pyplot(fig)

    # =========================
    # Classification Report
    # =========================

    st.subheader("Classification Report")

    report = classification_report(y, y_pred)

    st.text(report)

else:
    st.write("Upload dataset to continue")
