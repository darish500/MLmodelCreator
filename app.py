import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import zipfile
import os
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ==============================
# Streamlit App
# ==============================
st.title("🧠 AI Agent: Multi-Model Training & Explainability")
st.markdown("Upload your dataset, choose models, train, evaluate, and download!")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select target column (for supervised models)", df.columns)

    # Features/Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data
    test_size = st.slider("Test size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Select Models
    st.subheader("⚙️ Model Selection")
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KMeans (Unsupervised)": KMeans(n_clusters=2, random_state=42)
    }
    selected_models = st.multiselect("Choose models (supervised + unsupervised)", list(models_dict.keys()))

    epochs = st.number_input("Epochs / Iterations (for iterative models)", min_value=10, max_value=1000, value=100, step=10)

    train_btn = st.button("🚀 Train Models")

    if train_btn and selected_models:
        trained_models = {}
        metrics_results = []

        progress = st.progress(0)
        for i, model_name in enumerate(selected_models):
            model = models_dict[model_name]

            # If supervised
            if model_name != "KMeans (Unsupervised)":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                metrics_results.append([model_name, acc, prec, rec, f1])

                # Confusion Matrix
                st.subheader(f"📉 Confusion Matrix - {model_name}")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(fig)

                # SHAP explainability (only for tree/logistic)
                try:
                    st.subheader(f"🔎 SHAP Explanation - {model_name}")
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test)
                    shap.summary_plot(shap_values, X_test, show=False)
                    st.pyplot(bbox_inches="tight")
                except Exception as e:
                    st.warning(f"SHAP not available for {model_name}: {e}")

            else:
                # Unsupervised KMeans
                model.fit(X)
                labels = model.labels_
                df["Cluster"] = labels
                st.subheader("📊 KMeans Clustering Results")
                st.dataframe(df.head())

            # Save model
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            joblib.dump(model, model_filename)
            trained_models[model_name] = model_filename

            progress.progress((i + 1) / len(selected_models))

        # Metrics Table
        if metrics_results:
            st.subheader("📑 Model Performance")
            metrics_df = pd.DataFrame(metrics_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
            st.dataframe(metrics_df)

        # Stacking Ensemble (if >1 supervised models)
        supervised_selected = [m for m in selected_models if m != "KMeans (Unsupervised)"]
        if len(supervised_selected) > 1:
            st.subheader("🌀 Stacking Ensemble")
            estimators = [(name, models_dict[name]) for name in supervised_selected]
            stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
            stack_clf.fit(X_train, y_train)
            y_pred_stack = stack_clf.predict(X_test)
            st.write("Ensemble Accuracy:", accuracy_score(y_test, y_pred_stack))
            joblib.dump(stack_clf, "StackingEnsemble.pkl")

        # Download section
        st.subheader("📥 Download Trained Models")
        for name, file in trained_models.items():
            with open(file, "rb") as f:
                st.download_button(f"Download {name}", f, file_name=file)

        # Zip all models
        if trained_models:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for file in trained_models.values():
                    zipf.write(file)
            st.download_button("Download All Models (ZIP)", data=zip_buffer.getvalue(), file_name="all_models.zip")

