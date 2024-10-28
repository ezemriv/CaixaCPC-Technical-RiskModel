import os

class CONFIG:

    # Paths
    DATA_DIR = os.path.join('./', 'data')
    DATA_PATH = os.path.join(DATA_DIR, 'datos_ml_caixabank.csv')
    RESULTS_DIR = os.path.join('./', 'results')
    
    # Target and Features
    TARGET = 'CreditStatus'
    CAT_COLS = [
                'PaymentHistory',
    ]
    NUM_COLS = [
                'Age', 
                'MonthlyIncome', 
                'TotalDebt', 
                'CreditLimit',
                'NumberOfCreditCards',
                'YearsWithCompany',
                'AverageMonthlyBalance',
    ]

    # Model and Training Parameters
    SEED = 23
    N_FOLDS = 5
