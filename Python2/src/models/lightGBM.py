import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
from src.config import CONFIG
from src.data_processing import DataProcessor
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate

from src.config import CONFIG
import time

def train_evaluate_lgb_model(X_train, X_test, y_train, y_test, cat_cols, num_cols):
    """Train and evaluate a LightGBM Regression model, output results, and create SHAP plots."""


    lgb_params = {
                        'objective': 'binary',
                        'verbosity': -1,
                        # 'n_estimators': 100,
                        'boosting_type': 'gbdt',
                        'random_state': CONFIG.SEED,
                        }


    lgb_model = LGBMClassifier(**lgb_params)

    # Cross-validation setup
    scoring = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}

    cv_results = cross_validate(
        lgb_model, X_train, y_train, 
        cv=CONFIG.N_FOLDS, 
        scoring=scoring, 
        return_train_score=True
    )

    # Print cross-validation results with titles for clarity
    print("\n===== LGBM Model Training =====")
    print(f"  Train Accuracy: {np.mean(cv_results['train_accuracy']):.4f}")
    print(f"  Train ROC AUC: {np.mean(cv_results['train_roc_auc']):.4f}")
    print(f"  Cross-validation Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"  Cross-validation ROC AUC: {np.mean(cv_results['test_roc_auc']):.4f}")

    # Fit the pipeline to the training data
    start_time = time.time()
    lgb_model.fit(X_train, y_train)
    end_time = time.time()

    y_test_pred = lgb_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

    print("\n===== LGBM Model Test Results =====\n")

    print("*"*30)
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    print("*"*30)

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=['Good', 'Bad']);
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig(os.path.join(CONFIG.RESULTS_DIR, 'lgb_confusion_matrix.png'), dpi=120)
    plt.close()

    # Use TreeExplainer for LightGBM model
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(CONFIG.RESULTS_DIR, 'lgb_shap_summary.png'), dpi=120)
    plt.close()

    # Save metrics to file
    with open(os.path.join(CONFIG.RESULTS_DIR, 'lgbm_metrics.txt'), 'w') as f:
        f.write(f"Train Cross-validation Accuracy: {np.mean(cv_results['train_accuracy']):.4f}\n")
        f.write(f"Train Cross-validation ROC AUC: {np.mean(cv_results['train_roc_auc']):.4f}\n")
        f.write(f"Valid Cross-validation Accuracy: {np.mean(cv_results['test_accuracy']):.4f}\n")
        f.write(f"Valid Cross-validation ROC AUC: {np.mean(cv_results['test_roc_auc']):.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test ROC AUC: {test_auc:.4f}\n")
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")