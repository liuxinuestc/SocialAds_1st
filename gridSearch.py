import numpy as np
import pandas as pd
from sklearn import model_selection
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

#Import training and test data
train = pd.read_csv("train.csv")#.fillna(value=-999.0)
test = pd.read_csv("test.csv")#.fillna(value=-999.0)

# Encode variables
y_train = train.label
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

# XGBoost - sklearn method
gbm = xgb.XGBRegressor()

xgb_params = {
'learning_rate': [0.01, 0.1],
'n_estimators': [2000],
'max_depth': [3, 5, 7, 9],
# 'gamma': [0, 1],
'subsample': [0.6, 0.8],
'colsample_bytree': [0.5, 0.6, 0.7]
}

fit_params = {
'early_stopping_rounds': 30,
'eval_metric': 'rmse',
'eval_set': [[x_train,y_train]]
}

grid = model_selection.GridSearchCV(gbm, xgb_params, cv=5, fit_params=fit_params)
grid.fit(x_train,y_train)
