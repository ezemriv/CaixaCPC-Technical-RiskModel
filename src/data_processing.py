import pandas as pd
import numpy as np
from .config import CONFIG

class DataProcessor(CONFIG):
    
    def __init__(self):
        self.NUM_COLS = CONFIG.NUM_COLS
        self.CAT_COLS = CONFIG.CAT_COLS

    def load_data(self, path):

        df = pd.read_csv(path, sep = ";")
        df.set_index('CustomerID', inplace=True)

        # # set data types for memory efficiency
        df[self.NUM_COLS] = df[self.NUM_COLS].astype(np.int16)
        df[self.CAT_COLS] = df[self.CAT_COLS].astype('category')

        return df
    
    def clean_data(self, df):
        
        # Convert target to boolean
        df[CONFIG.TARGET] = np.where(df[CONFIG.TARGET] == 'Bad', 1, 0).astype(np.int8)

        return df

    def feature_engineering(self, df):
        
        # Convert NumberOfCreditCards to categorical
        df['NumberOfCreditCards_cat'] = df['NumberOfCreditCards'].astype('category')

        # Added new features
        df['income_above_mean'] = np.where(df['MonthlyIncome'] > df['MonthlyIncome'].mean(), 1, 0)
        df['debts_above_mean'] = np.where(df['TotalDebt'] > df['TotalDebt'].mean(), 1, 0)
        df['imcome_debt_ratio'] = df['MonthlyIncome'] / (df['TotalDebt'] + 1e-6)
        df['debt_credit_ratio'] = df['TotalDebt'] / (df['CreditLimit'] + 1e-6)
        df['high_credit_utilization'] = np.where(df['debt_credit_ratio'] > 0.3, 1, 0)
        df['years_with_company_cat'] = pd.cut(df['YearsWithCompany'], bins=[0, 5, 10, 15, np.inf], labels=['New', 'Intermediate', 'Long Term', 'Very Long Term'])
        df['debt_income_category'] = pd.cut(df['imcome_debt_ratio'], bins=[-np.inf, 0.1, 0.3, np.inf], labels=['Low', 'Moderate', 'High'])
        df['income_debt_interaction'] = df['MonthlyIncome'] * df['TotalDebt']
        df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, np.inf], labels=['Young', 'Middle-aged', 'Senior'])

        # Categorical columns
        new_cat_cols = ['NumberOfCreditCards_cat', 
                        'years_with_company_cat', 
                        'debt_income_category', 
                        'age_group'
                        ]
        
        # Numerical columns
        new_num_cols = [
            'income_above_mean', 
            'debts_above_mean', 
            'imcome_debt_ratio', 
            'debt_credit_ratio', 
            'high_credit_utilization', 
            'income_debt_interaction'
        ]

        return df, new_cat_cols, new_num_cols

    def process_data(self, path):

        df = self.load_data(path)
        df = self.clean_data(df)
        df, new_cat_cols, new_num_cols = self.feature_engineering(df)

        self.NUM_COLS = self.NUM_COLS + new_num_cols
        self.CAT_COLS = self.CAT_COLS + new_cat_cols

        return df