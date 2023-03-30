#%%
import numpy as np
import pandas as pd

from preprocess import get_data
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
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
  ('ET', ExtraTreesRegressor()),

  #('XGBRF', XGBRFRegressor()),
  #('XGBR', XGBRegressor()),
  ('MLP', MLPRegressor(solver='lbfgs', max_iter=1000)),
]

pipelines = spot_check(a)

#%%
x_train, x_val, y_train, y_val = get_data()

#%%
#%%

names, results = [], []
for name, model in pipelines:
  cv = KFold(n_splits=10)
  #cv_results = cross_val_score(model, x_train.values, y_train.ravel(),cv=cv, scoring=rmse_scorer)
  cv_results = cross_val_score(model, x_train, y_train, cv=10, scoring='r2')

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}")



# %%
x_test_data, idd = get_data(False)
x_test = x_test_data.reindex(columns=x_train.columns, fill_value=0)

#%%
model = RandomForestRegressor()
model.fit(x_train, y_train)

pred = model.predict(x_test)
#%%

data = {
  'Id': idd.values,
  'SalePrice': pred
}

sol = pd.DataFrame(data)
sol.to_csv('summit.csv', index=False)

#r2_score(y_val.values, pred)