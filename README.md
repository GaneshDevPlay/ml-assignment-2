# Machine Learning Assignment 2: Classification Models & Deployment

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five supervised Machine Learning classification algorithms on a multi-feature tabular dataset. The model artifacts are saved and integrated into an interactive web application built with Streamlit and deployed on Streamlit Community Cloud. The system enables users to upload test datasets, select classification models, and view evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, and MCC) alongside confusion matrices and classification reports without retraining.

## b. Dataset Description
- **Dataset Name**: UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **Instance Count**: 569 instances (Training split: 455, Test split: 114)
- **Feature Count**: 30 continuous numeric attributes derived from digitized cell nucleus images
- **Target Variable**: `target` (Binary Classification: `0` = Malignant, `1` = Benign)
- **Feature Attributes**: Mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## c. Repository & Application Links
- **GitHub Repository**: [https://github.com/GaneshDevPlay/ml-assignment-2](https://github.com/GaneshDevPlay/ml-assignment-2)
- **Live Streamlit App**: [https://ml-assignment-2-6y2je4vwbgrt4r4n6pksve.streamlit.app/](https://ml-assignment-2-6y2je4vwbgrt4r4n6pksve.streamlit.app/)

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## e. Observations & Analysis

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Exceptional overall accuracy (98.25%) and top-tier MCC score (0.9623). Standard scaling allows linear hyperplanes to effectively isolate linearly separable diagnostic features with minimal misclassifications. |
| **Decision Tree** | Yielded lowest overall metrics (Accuracy: 91.23%, MCC: 0.8174). Axis-aligned decision boundaries struggle with complex correlations among the 30 continuous features, leading to slight overfitting on training splits. |
| **KNN** | Strong non-parametric performance (Accuracy: 95.61%, AUC: 0.9788). Feature scaling enables effective Euclidean distance calculations in high dimensions, maintaining high precision and recall. |
| **Naive Bayes** | Solid probability calibration (AUC: 0.9878, Accuracy: 93.86%). Despite the strong independence assumption between features, Gaussian Naive Bayes provides robust baseline classifications. |
| **Support Vector Machine** | Tied for best predictive performance (Accuracy: 98.25%, AUC: 0.9950, F1: 0.9861). Radial Basis Function (RBF) kernel accurately constructs optimal decision hyperplanes with maximal margin separation. |
| **Overall Winner** | **Logistic Regression & Support Vector Machine (Tie)**. Logistic Regression is selected as the primary winner due to identical top performance (98.25% Accuracy, 0.9623 MCC) combined with lower computational complexity and better interpretability. |