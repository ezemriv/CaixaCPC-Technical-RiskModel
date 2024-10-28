import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

class InitialEDA:
    """Class for performing exploratory data analysis (EDA) on a DataFrame."""

    @staticmethod
    def plot_histograms(df, num_cols: list[str] = []):
        """Plots histograms for numeric variables in the DataFrame."""
        df[num_cols].hist(bins=15, figsize=(15, 10))
        plt.suptitle('Histograms of Numeric Variables')
        plt.show()
    
    @staticmethod
    def plot_histograms_classes(df, target: str, num_cols: list[str] = []):
        """Plots superimposed histograms for numeric variables, separated by the target classes."""

        target_classes = df[target].unique()
        n_cols = 2
        n_rows = round(len(num_cols) / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
        axs = axs.flatten()

        for i, col in enumerate(num_cols):
            for cls in target_classes:
                sns.histplot(df[df[target] == cls][col], 
                             label=f'{target} = {cls}', 
                             ax=axs[i], 
                             bins=15, 
                             element="step")
            axs[i].set_title(f'Histogram of {col} by {target}')
            axs[i].legend(title=target)
        
        # Hide any unused subplots
        for j in range(len(num_cols), len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_barplots_normalized(df, target: str, exclude: list[str] = [], cat_cols: list[str] = []):
            """Plots normalized bar plots for categorical variables, separated by target classes."""
            
            categorical_cols = [col for col in cat_cols if col not in exclude]
            n_cols = 2
            n_rows = round(len(categorical_cols) / n_cols)
            
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, n_rows * 4))
            axs = axs.flatten()

            for i, col in enumerate(categorical_cols):
                # Get normalized value counts for each class of the target
                data = df.groupby(target)[col].value_counts(normalize=True).rename("percentage").reset_index()
                sns.barplot(data=data, x="percentage", y=col, hue=target, ax=axs[i])
                axs[i].set_title(f'Normalized Value Counts of {col} by {target}')
                axs[i].set_xlabel('Normalized Value Counts')
                axs[i].set_ylabel('Categories')

            for i in range(len(categorical_cols), len(axs)):
                axs[i].axis('off')

            plt.tight_layout()
            plt.show()