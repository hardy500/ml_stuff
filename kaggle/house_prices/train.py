#%%
import numpy as np
import pandas as pd

from preprocess import get_data
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


from sklearn.ensemble import (
    AdaBoostRegressor,  GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor, XGBRFRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_regression

import matplotlib.pyplot as plt


#%%
def spot_check(a):
  c = []
  for i in range(len(a)):
    c.append(a[i])
  return c

a = [
  #('LS', Lasso()),
  #('KNN', KNeighborsRegressor()),
  #('DT', DecisionTreeRegressor()),
  #('SVR', SVR()),

  #('ADA', AdaBoostRegressor()),
  #('GB', GradientBoostingRegressor()),
  ('RF', RandomForestRegressor()),
  ('RID', Ridge()),
  ('ET', ExtraTreesRegressor()),

  #('XGBRF', XGBRFRegressor()),
  #('XGBR', XGBRegressor()),
  #('MLP', MLPRegressor(solver='adam', max_iter=1000))
]

pipelines = spot_check(a)

#%%
x_train, x_val, y_train, y_val = get_data()
#%%

fs = SelectKBest(score_func=f_regression, k='all')
fs.fit(x_train, y_train.ravel())
x_train_fs = fs.transform(x_train)

scores = []
for i in range(len(fs.scores_)):
  if f"{fs.scores_[i]}" != 'nan':
    scores.append((fs.scores_[i], i))

scores = sorted(scores, key=lambda x: x[0], reverse=True)
features = [i[1] for i in scores]
#%%
x_train = x_train.iloc[:,features[:]]
y_train = np.log1p(y_train)

#%%

#%%
#%%

names, results = [], []
for name, model in pipelines:
  cv = KFold(n_splits=10)
  cv_results = cross_val_score(model, x_train, y_train, cv=cv, scoring='r2')

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}")


#%%
from hyperopt import fmin, tpe, hp
space = {
    'n_estimators': hp.choice('n_estimators', range(10, 200)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 11))
}

def objective(params):
  model = RandomForestRegressor(**params)
  score = -1.0 * cross_val_score(model,x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
  return score

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
#%%
best


# %%
x_test_data, idd = get_data(False)
x_test = x_test_data.reindex(columns=x_train.columns, fill_value=0)

#%%

best_params = {
    'n_estimators': best['n_estimators'] + 10,
    'max_depth': best['max_depth'] + 1,
    'min_samples_split': best['min_samples_split'] + 2
}

model = RandomForestRegressor(**best_params)
model.fit(x_train, y_train)

pred = model.predict(x_test)

#%%

data = {
  'Id': idd.values,
  'SalePrice': np.exp(pred)
}

sol = pd.DataFrame(data)
sol.to_csv('summit.csv', index=False)