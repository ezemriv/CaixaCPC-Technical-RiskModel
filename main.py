import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import CONFIG
from src.data_processing import DataProcessor
from src.models.logistic_regression import train_evaluate_logistic_regression
from src.models.lightGBM import train_evaluate_lgb_model

def main():
    # Initialize DataProcessor
    dp = DataProcessor()

    # Load and preprocess data
    data = dp.process_data(CONFIG.DATA_PATH)

    # Combine original and newly engineered categorical columns
    cat_cols = dp.CAT_COLS.copy()
    num_cols = dp.NUM_COLS.copy()
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[CONFIG.TARGET])
    y = data[CONFIG.TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=CONFIG.SEED, stratify=y)
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Call the logistic regression training and evaluation function
    train_evaluate_logistic_regression(X_train, X_test, y_train, y_test, cat_cols, num_cols)

    # Call the LGBM training and evaluation function
    train_evaluate_lgb_model(X_train, X_test, y_train, y_test, cat_cols, num_cols)

if __name__ == "__main__":
    main()
