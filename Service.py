# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:43:00 2024

@author: morin
"""

columns_informations = {
    'EQT_S&P500_USD':'^GSPC',
    'EQT_Euronext100_Index_EUR':'^N100',
    'EQT_FTSE100_GBP':'^FTSE',
    'EQT_CAC40_EUR':'^FCHI',
    'EQT_DAX30_EUR':'^GDAXI',
    'EQT_Nikkei225_JPY':'^N225',
    'EQT_VIX_Index_USD':'^VIX',
    'FX_Treasury_Yield_5yo_USD':'^FVX',
    'FX_CBOE_Interest_Rate_10yo_USD':'^TNX',
    'RealEstate_SPDR_Fund_USD':'XLRE',
    'Commo_OIL_USD':'CL=F',
    'Commo_NaturalGas_USD':'NG=F',
    'Commo_Gold_USD':'GC=F',
    'Forex_USDEUR':'EUR=X',
    'Forex_USDGBP':'GBP=X',
    'Forex_USDJPY':'JPY=X',
    }

param_grid_kmeans = {
    'n_clusters': [2, 3, 4,],
    'init': ['k-means++', 'random'],
    'max_iter': [300, 400, 500, 1000, 2000],
    'n_init': [40, 50, 100],
    'algorithm': ["lloyd", "elkan"],
    'tol': [0.0001, 0.00001]
}

param_grid_xgboost = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'objective': ['multi:softmax'],
    'num_class': [4]
}

