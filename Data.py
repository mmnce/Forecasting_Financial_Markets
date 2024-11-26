# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:42:08 2024

@author: morin
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from functools import reduce
from Service import columns_informations
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA

class DataManagement:
    
    def __init__(self):
        self.start_date = "2015-01-01"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.max_attempts = 5
        self.filepath_save = "C:/Users/morin/Desktop/Python_Project/Forecasting_Financial_Market/Database/"
    
    def get_historical_data(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date)
        data = data.dropna()
        data = data.reset_index()
        data = data[["Date","Adj Close"]]
        return data
    
    def arrange_dataset(self, df):
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        return df
    
    def set_historicalprice_dataset(self):
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            df_list = []
            for key, value in columns_informations.items():
                ticker = value
                columns_name = key
                try:
                    data = self.get_historical_data(ticker)
                    data = data.rename(columns={'Adj Close': columns_name})
                    df_list.append(data)
                except Exception as e:
                    print(f"Data recovery problem for the ticker : {ticker}. Erreur : {e}")
        
                if len(df_list) == len(columns_informations.values()):
                    dataset = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), df_list)
                    print(f"Complete dataset created after {attempt} attempt(s)")
                    return dataset
            print(f"Tentative {attempt} échouée. Re-essai...")
        print("Maximum number of attempts reached. Impossible to build the complete dataset")
        return None
    
    def transform_USD(self, df):
        for column in df.columns:
            if column not in ['Forex_USDEUR', 'Forex_USDGBP', 'Forex_USDJPY']:
                new_column = column + "_USD"
                if "EUR" in column:
                    df[new_column] = df[column] * (1/df['Forex_USDEUR'])
                    df = df.drop(columns=[column])
                elif "JPY" in column:
                    df[new_column] = df[column] * (1/df['Forex_USDJPY'])
                    df = df.drop(columns=[column])
                elif "GBP" in column:
                    df[new_column] = df[column] * (1/df['Forex_USDGBP'])
                    df = df.drop(columns=[column])
        return df
            
    
    def save_dataset(self):
        filepath_dataset = self.filepath_save + "Dataset.xlsx"
        dataset = self.set_historicalprice_dataset()
        dataset = self.arrange_dataset(dataset)
        dataset = self.transform_USD(dataset)
        dataset.to_excel(filepath_dataset)
    
    def get_dataset(self, file_name):
        file_path = self.filepath_save + file_name + ".xlsx"
        df = pd.read_excel(file_path)
        df = df.set_index('Date')
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors='coerce').astype('float64')
        return df 
    
    def calculate_log_returns(self, df):
        df = df[~(df < 0).any(axis=1)]
        df_log_returns = np.log(df / df.shift(1))
        df_log_returns = df_log_returns.dropna()
        df_log_returns = df_log_returns.sort_index()
        return df_log_returns
    
    def save_daily_return(self):
        filepath_daily_return = self.filepath_save + "Daily_return.xlsx"
        dataset = self.get_dataset("Dataset")
        df_log_returns = self.calculate_log_returns(dataset)
        df_log_returns.to_excel(filepath_daily_return)

class DataAnalytics:
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.root = tk.Tk()
        self.root.title("Data Analytics")
        self.label = tk.Label(self.root, text="Select a column")
        self.label.pack(padx=10, pady=10)
        self.combo = ttk.Combobox(self.root, values=list(self.dataset.columns))
        self.combo.pack(padx=10, pady=10)
        self.button = tk.Button(self.root, text="Plot Daily Return", command=self.plot_return)
        self.button.pack(padx=10, pady=10)
        self.corr_button = tk.Button(self.root, text="Plot Correlation Matrix", command=self.corr_matrix)
        self.corr_button.pack(padx=10, pady=10)
        self.norm_button = tk.Button(self.root, text="Plot Normality", command=self.plot_normality)
        self.norm_button.pack(padx=10, pady=10)
    
    def plot_return(self):
        column_name = self.combo.get()
        plt.figure(figsize=(10, 6))
        self.dataset[column_name].plot(title=f"Daily Return of {column_name}")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.show()
    
    def corr_matrix(self):
        correlation_matrix = self.dataset.corr()
        plt.figure(figsize=(16, 14))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png", format="png", dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_normality(self):
        column_name = self.combo.get()
        mean, std_dev = self.dataset[column_name].mean(), self.dataset[column_name].std()
        plt.figure(figsize=(10, 6))
        sns.histplot(self.dataset[column_name], bins=30, kde=False, stat='density', color="skyblue", label='Data Distribution')
        x = np.linspace(self.dataset[column_name].min(), self.dataset[column_name].max(), 100)
        plt.plot(x, norm.pdf(x, mean, std_dev), color='red', label='Normal Distribution')
        plt.title(f"Normality Check for {column_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
    
    def plot_pca_elbow(self):
        pca = PCA()
        pca.fit(self.dataset)
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', color='b')
        plt.xlabel("Number of Principle Components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Elbow method for PCA")
        plt.show()
    
    def perform_pca(self, n_components):
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(self.dataset)
        pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)])
    
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, color='blue')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA")
            plt.grid(True)
            plt.show()
    
        elif n_components == 3:
            # Représentation en 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.5, color='blue')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("Graphical representation of our dataset using a 3-principal component PCA")
            plt.show()

    def clustering_analysis(self, dataset_clustering):
        grouped_df = dataset_clustering.groupby('Macroeconomics_Regime')
        mean_df = grouped_df.mean()
        median_df = grouped_df.median()
        std_df = grouped_df.std()
        mean_df.to_excel("mean_cluster.xlsx")

    def analyse_forecasting_results(self, df_gridsearchcv_results, column):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_gridsearchcv_results, x=column, y="mean_test_score", marker="o")
        plt.title(f'Impact of {column} on Accuracy Score')
        plt.xlabel(column)
        plt.ylabel("Accuracy Score")
        plt.grid(True)
        plt.show()


        
        
        
        
        
      
