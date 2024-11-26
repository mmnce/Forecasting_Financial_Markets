# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:44:52 2024

@author: morin
"""
from Service import param_grid_kmeans, param_grid_xgboost
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, quantile_transform
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb



class Clustering:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None

    def preprocess_data(self, method="StandardScaler"):
        scaler_methods = {
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer": QuantileTransformer(),
            "quantile_transform": lambda data: quantile_transform(data)
        }
        scaler = scaler_methods.get(method, StandardScaler())
        if method == "quantile_transform":
            self.data_scaled = scaler(self.data)
        else:
            self.data_scaled = scaler.fit_transform(self.data)

    def silhouette_scorer(self, estimator, X):
        labels = estimator.predict(X)
        return silhouette_score(X, labels)

    def perform_gridsearch(self):
        self.preprocess_data("RobustScaler")
        model = KMeans(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_kmeans,
                                   cv=2, scoring=self.silhouette_scorer)
        grid_search.fit(self.data_scaled)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        results = grid_search.cv_results_
        results_df = pd.DataFrame(results)
        results_df.to_excel("grid_search_results.xlsx", index=False)
        return best_estimator, best_params, best_score

    def manual_grid_search(self):
        grid_params = pd.read_excel("manual_grid_search.xlsx")
        inertia, silhouette_scores = [], []
        for i in range(0, len(grid_params), 1):
            preprocess_method = grid_params.loc[i, "preprocess_method"]
            self.preprocess_data(preprocess_method)
            model = KMeans(n_clusters=int(grid_params.loc[i, "param_n_clusters"]), init=grid_params.loc[i, "param_init"],
                           n_init=grid_params.loc[i, "param_n_init"], max_iter=grid_params.loc[i, "param_max_iter"],
                           algorithm=grid_params.loc[i, "param_algorithm"], random_state=42)
            model.fit(self.data_scaled)
            inertia.append(model.inertia_)
            silhouette_avg = silhouette_score(self.data_scaled, model.labels_)
            silhouette_scores.append(silhouette_avg)
        grid_params['Inertia'] = inertia
        grid_params['Silhouette_Score'] = silhouette_scores
        return grid_params

    def fit(self, preprocess_method:str, n_clusters:int, init:str, n_init:int, max_iter:int, algorithm:str, tol:float):
        self.preprocess_data(preprocess_method)
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, algorithm=algorithm,
                            tol=tol, random_state=42)
        self.labels_ = self.model.fit_predict(self.data_scaled)
        self.data['Macroeconomics_Regime'] = self.labels_
        self.data.to_excel("Dataset_Clustering.xlsx")
        self.cluster_centers_ = self.model.cluster_centers_
        silhouette_avg = silhouette_score(self.data_scaled, self.labels_)
        print(self.cluster_centers_)
        print(f'Silhouette Score: {silhouette_avg}')

    def plot_clusters(self, pca_shape):
        pca = PCA(n_components=pca_shape)
        pca_data = pca.fit_transform(self.data_scaled)
        pca_df = pd.DataFrame(pca_data, columns=[f"PC{i + 1}" for i in range(pca_shape)])
        if pca_shape == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=self.labels_, cmap='viridis', alpha=0.6)
            plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], s=30, c='red', label='Centroids')
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Clustering K-means (2D)")
            plt.legend()
            plt.show()
        elif pca_shape == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], pca_df.iloc[:, 2], c=self.labels_, cmap='viridis',
                       alpha=0.6)
            ax.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], self.cluster_centers_[:, 2], s=200,
                       c='red', label='Centroids')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title("Clustering K-means (3D)")
            plt.legend()
            plt.show()
        else:
            print("La visualisation des clusters est disponible uniquement pour des données à 2 ou 3 dimensions.")

    def elbow_and_silhouette_method(self, max_clusters=10):
        self.preprocess_data()
        inertia = []
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.data_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        k_values = list(range(2, max_clusters + 1))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(k_values, inertia, marker='o', color="blue")
        ax1.set_xlabel("Nombre de clusters")
        ax1.set_ylabel("Inertie (Méthode du coude)")
        ax1.set_title("Méthode du coude pour déterminer le nombre optimal de clusters")
        ax1.grid(True)
        ax2.plot(k_values, silhouette_scores, marker='s', color="green")
        ax2.set_xlabel("Nombre de clusters")
        ax2.set_ylabel("Score de silhouette")
        ax2.set_title("Score de silhouette pour déterminer le nombre optimal de clusters")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()


class Forecasting:

    def __init__(self, dataset_clusters):
        self.dataset_clusters = dataset_clusters
        self.Y = self.dataset_clusters['Macroeconomics_Regime']
        self.X = self.dataset_clusters.drop('Macroeconomics_Regime', axis=1)

    def preprocess_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def grid_search(self):
        model = xgb.XGBClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_xgboost, scoring='accuracy', cv=3, verbose=1)
        grid_search.fit(self.X, self.Y)
        results = grid_search.cv_results_
        results_df = pd.DataFrame(results)
        results_df.to_excel("xgboost_grid_search_results.xlsx", index=False)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_estimator, best_params, best_score

    def model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(self.Y.unique()), eval_metric="mlogloss")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))






