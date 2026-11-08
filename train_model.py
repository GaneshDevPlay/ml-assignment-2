import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def main():
    # 1. Create model directory
    os.makedirs('model', exist_ok=True)
    
    # 2. Load UCI / Kaggle Breast Cancer Diagnostic Dataset
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Standard Scaling for distance/gradient-sensitive models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Feature Scaler
    joblib.dump(scaler, 'model/scaler.joblib')
    
    # Save test dataset CSV for Streamlit web app evaluation
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print(" Saved 'test_data.csv' successfully.")
    
    # 5. Model Initialization
    models = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), True),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
        'KNN': (KNeighborsClassifier(n_neighbors=5), True),
        'Naive Bayes': (GaussianNB(), False),
        'Support Vector Machine': (SVC(probability=True, random_state=42), True)
    }
    
    model_filenames = {
        'Logistic Regression': 'model/logistic_regression.joblib',
        'Decision Tree': 'model/decision_tree.joblib',
        'KNN': 'model/knn.joblib',
        'Naive Bayes': 'model/naive_bayes.joblib',
        'Support Vector Machine': 'model/support_vector_machine.joblib'
    }
    
    metrics_summary = []
    
    # 6. Model Training & Evaluation Loop
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        # Fit model
        model.fit(X_tr, y_train)
        
        # Save model artifact
        joblib.dump(model, model_filenames[name])
        
        # Inference
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        metrics_summary.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })
    
    summary_df = pd.DataFrame(metrics_summary)
    print("\n Model Performance Summary:")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()