import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from src.config import CONFIG
import time

def train_evaluate_logistic_regression(X_train, X_test, y_train, y_test, cat_cols, num_cols):
    """Train and evaluate a Logistic Regression model, output results, and create SHAP plots."""
    
    # Build preprocessing pipeline (scaling and encoding)
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),                  # Standard scaling for numerical features
        (OneHotEncoder(drop='first'), cat_cols)               # One-hot encoding for categorical features
    )

    # Create a pipeline with preprocessing and logistic regression
    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(random_state=CONFIG.SEED, max_iter=1000)  # Logistic regression model
    )

    # Cross-validation settings
    scoring = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}

    # Perform cross-validation
    cv_results = cross_validate(pipeline, X_train, y_train, 
                                cv=CONFIG.N_FOLDS, 
                                scoring=scoring, 
                                return_train_score=True)

    # Print cross-validation results with titles for clarity
    print("\n===== Logistic Regression Model Training =====")
    print(f"  Train Accuracy: {np.mean(cv_results['train_accuracy']):.4f}")
    print(f"  Train ROC AUC: {np.mean(cv_results['train_roc_auc']):.4f}")
    print(f"  Cross-validation Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"  Cross-validation ROC AUC: {np.mean(cv_results['test_roc_auc']):.4f}")

    # Fit the pipeline to the training data
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()

    # Evaluate the model on the test set
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    print("\n===== Logistic Regression Model Test Results =====\n")

    print("*"*30)
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    print("*"*30)
    
    # Confusion Matrix Display
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=['Good', 'Bad'])
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig(os.path.join(CONFIG.RESULTS_DIR, 'lr_confusion_matrix.png'), dpi=120)
    plt.close()
    
    # SHAP explainability for Logistic Regression
    log_reg_model = pipeline.named_steps['logisticregression']

    # Preprocess training and test sets for SHAP analysis
    X_train_ = preprocessor.fit_transform(X_train)
    X_test_ = preprocessor.transform(X_test)

    # Initialize SHAP explainer
    explainer = shap.Explainer(log_reg_model, X_train_)
    shap_values = explainer(X_test_)

    # Retrieve feature names (numerical + encoded categorical features)
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(cat_cols)
    feature_names = np.hstack([num_cols, cat_features])

    # SHAP summary plot with correct feature names
    shap.summary_plot(shap_values, X_test_, feature_names=feature_names, max_display=10, show=False)
    plt.savefig(os.path.join(CONFIG.RESULTS_DIR, 'lr_shap_summary.png'))
    plt.close()

    # Save metrics to file
    with open(os.path.join(CONFIG.RESULTS_DIR, 'lr_metrics.txt'), 'w') as f:
        f.write(f"Train Cross-validation Accuracy: {np.mean(cv_results['train_accuracy']):.4f}\n")
        f.write(f"Train Cross-validation ROC AUC: {np.mean(cv_results['train_roc_auc']):.4f}\n")
        f.write(f"Valid Cross-validation Accuracy: {np.mean(cv_results['test_accuracy']):.4f}\n")
        f.write(f"Valid Cross-validation ROC AUC: {np.mean(cv_results['test_roc_auc']):.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test ROC AUC: {test_auc:.4f}\n")
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
