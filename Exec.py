# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:48:44 2024

@author: morin
"""

from Data import DataManagement, DataAnalytics, pd
from Model import Clustering, Forecasting

def save_data():
    DataManagement().save_dataset()
    DataManagement().save_daily_return()

def analysis():
    df = DataManagement().get_dataset("Daily_return")
    Analyser = DataAnalytics(df)
    Analyser.root.mainloop()
    Analyser.plot_pca_elbow()
    Analyser.perform_pca(3)

def train_clusters():
    df = DataManagement().get_dataset("Daily_return")
    clustering = Clustering(df)
    clustering.elbow_and_silhouette_method()
    best_estimator, best_params, best_score = clustering.perform_gridsearch()
    print(f'best estimator : {best_estimator}')
    print(f'best params : {best_params}')
    print(f'best score : {best_score}')

def clustering():
    df = DataManagement().get_dataset("Daily_return")
    clustering = Clustering(df)
    # Hyperparameters training thanks to a manual grid 
    grid_params = clustering.manual_grid_search()
    grid_params.to_excel("manual_grid_search_results.xlsx")
    # The optimal combination of hyperparameters in the xgboost model in relation to our data
    clustering.fit(preprocess_method="RobustScaler", n_clusters=4, init="k-means++", n_init=1000,
                   max_iter=10000, algorithm="lloyd", tol=0.0001)
    # Graphical representation in 2 or 3 dimensions of our different clusters
    clustering.plot_clusters(3)

def clusters_analysis():
    df = pd.read_excel("Dataset_Clustering.xlsx")
    Analyser = DataAnalytics(df)
    Analyser.clustering_analysis(df)

def forecasting_clusters():
    df = DataManagement().get_dataset("Dataset_Clustering")
    forecasting = Forecasting(df)
    forecasting.grid_search()

def forecasting_analysis():
    results_df = pd.read_excel("xgboost_grid_search_results.xlsx")
    Analyser = DataAnalytics(results_df)
    Analyser.analyse_forecasting_results(results_df, "param_max_depth")

# Step 1
save_data()
# Step 2
analysis()
# Step 3
clustering()
# Step 4
clusters_analysis()
# Step 5
forecasting_clusters()
# Step 6
forecasting_analysis()
