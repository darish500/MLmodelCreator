import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC

import joblib

# ---------------- Streamlit UI ---------------- #
st.title("📊 ML Model Creator & Trainer")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:", data.head())

    # Handle categorical features
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # Handle NaNs
    data = data.fillna(0)

    # Select target
    target_col = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Ensure y is numeric
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Train-test split
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model options
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    selected_models = st.multiselect("Select Models to Train", list(models_dict.keys()))

    epochs = st.number_input("Training Epochs (for Gradient Boosting only)", min_value=50, max_value=500, step=50, value=100)

    if st.button("Train Models"):
        results = {}
        trained_models = {}

        for model_name in selected_models:
            model = models_dict[model_name]

            # Set epochs if GradientBoosting
            if model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=epochs)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            trained_models[model_name] = model

            st.write(f"### {model_name}")
            st.write("Accuracy:", acc)
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Stacking Ensemble
        if len(selected_models) > 1:
            estimators = [(name, trained_models[name]) for name in trained_models]
            stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500))
            stack_model.fit(X_train, y_train)
            y_pred = stack_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write("### 🧩 Stacking Ensemble")
            st.write("Accuracy:", acc)

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Save model
            joblib.dump(stack_model, "stacking_model.pkl")
            st.download_button("Download Stacking Model", data=open("stacking_model.pkl", "rb"), file_name="stacking_model.pkl")

