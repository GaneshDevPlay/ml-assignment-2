import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classifier Evaluation Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Machine Learning Classification Dashboard")
st.markdown("Upload test dataset CSV, select pre-trained classification models, and analyze performance metrics.")

# Models configuration
MODEL_CONFIG = {
    'Logistic Regression': {'file': 'model/logistic_regression.joblib', 'scaled': True},
    'Decision Tree': {'file': 'model/decision_tree.joblib', 'scaled': False},
    'KNN': {'file': 'model/knn.joblib', 'scaled': True},
    'Naive Bayes': {'file': 'model/naive_bayes.joblib', 'scaled': False},
    'Support Vector Machine': {'file': 'model/support_vector_machine.joblib', 'scaled': True}
}

# Sidebar - Dataset Upload and Model Selection
st.sidebar.header("📁 Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"],
    help="CSV file must contain all feature columns and the 'target' column."
)

selected_model_name = st.sidebar.selectbox(
    "Select Classifier Model",
    list(MODEL_CONFIG.keys())
)

st.sidebar.info("Models loaded directly from disk (`model/`) without retraining.")

if uploaded_file is not None:
    # Load dataset
    df_test = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded Test Dataset Preview")
    st.dataframe(df_test.head(10), use_container_width=True)
    
    if 'target' not in df_test.columns:
        st.error("Error: Target column 'target' not found in uploaded dataset.")
    else:
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']
        
        try:
            # Load scaler and model
            scaler = joblib.load('model/scaler.joblib')
            model_info = MODEL_CONFIG[selected_model_name]
            model = joblib.load(model_info['file'])
            
            # Apply scaling if required
            if model_info['scaled']:
                X_input = scaler.transform(X_test)
            else:
                X_input = X_test
            
            # Model Inference
            y_pred = model.predict(X_input)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_input)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_input)
            else:
                y_proba = y_pred
            
            # Metrics Calculation
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Section: Key Metrics Display
            st.markdown(f"### 🎯 Performance Metrics — {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.divider()
            
            # Section: Confusion Matrix and Classification Report
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📌 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=['Malignant (0)', 'Benign (1)'],
                            yticklabels=['Malignant (0)', 'Benign (1)'])
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
                
            with col_right:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)
                
            # Section: Predictions Output Preview
            with st.expander("🔍 View Detailed Instance Predictions"):
                results_df = X_test.copy()
                results_df['True Target'] = y_test
                results_df['Predicted Target'] = y_pred
                results_df['Probability Class 1'] = np.round(y_proba, 4)
                st.dataframe(results_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error during model inference: {e}")
else:
    st.info(" Please upload `test_data.csv` via the sidebar to execute evaluation.")