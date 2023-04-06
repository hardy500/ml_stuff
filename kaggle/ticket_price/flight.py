#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

#%%

df = pd.read_csv('data/flight.csv', low_memory=False)
df = df.loc[:,~df.columns.str.startswith('Unnamed')]
df.columns = [i.strip() for i in df.columns]

#%%
df['Number of  Stoppage'] = [int(i.split(' ')[0]) if i.split(' ')[0].isdigit() else 0 for i in df['Number of  Stoppage']]

#%%
def duration_to_minutes(duration):
  if '$' not in duration:
    td = pd.Timedelta(duration)
    return td.total_seconds()/60

df['duration_minutes'] = df['Travel  Time'].apply(duration_to_minutes)
df = df.drop('Travel  Time', axis=1)
df.rename(columns={
  'Ticket prize(Doller)':'ticket_price',
  'Number of  Stoppage': 'n_stoppage'
  }, inplace=True
)
#%%
df['ticket_price'].replace(['Alaska', '$'], np.nan, inplace=True)
df['ticket_price'].fillna(df['ticket_price'].median(), inplace=True)
df['Depreture  Airport'].fillna('bfill', inplace=True)
df['Destination Airport'].fillna('bfill', inplace=True)
df['duration_minutes'].fillna(df['duration_minutes'].median(), inplace=True)
df['Airline name'].replace("That's 6% off the retail price", np.nan, inplace=True)
df['Airline name'].fillna('bfill', inplace=True)
df['Arrival Time'].fillna('bfill', inplace=True)
df['Depreture Time'].fillna('bfill', inplace=True)
df['1st Stoppage Waiting Hour'].astype('str').replace('nan', np.nan, inplace=True)
df['1st Stoppage Waiting Hour'].fillna('bfill', inplace=True)

encoder = OrdinalEncoder()
df['Arrival Time'] = encoder.fit_transform(df['Arrival Time'].values.reshape((-1, 1)))
df['Depreture Time'] = encoder.fit_transform(df['Depreture Time'].values.reshape((-1, 1)))
df['1st Stoppage Waiting Hour'] = encoder.fit_transform(df['1st Stoppage Waiting Hour'].values.reshape((-1, 1)))

#%%

features = ['n_stoppage', 'duration_minutes', 'Depreture Time', 'Arrival Time', '1st Stoppage Waiting Hour']
y = np.log(df['ticket_price'].astype('float32'))
one_hot = pd.get_dummies(df[['Depreture  Airport', 'Destination Airport', 'Airline name']], drop_first=True)

scaler = MinMaxScaler()
x = pd.DataFrame(scaler.fit_transform(df[features]), columns=df[features].columns)
x = pd.concat([x, one_hot], axis=1)

#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=.95)
x = pca.fit_transform(x)

#%%

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

#%%
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor

def spot_check(a):
  c = []
  for i in range(len(a)):
    c.append(a[i])
  return c

a = [
  #('LR', LinearRegression()),
  #('XGF', XGBRFRegressor()),
  ('XGR', XGBRegressor()),
  ('KNN', KNeighborsRegressor()),
  ('BG', BaggingRegressor()),
  #('DT', DecisionTreeRegressor()),
  ('RF', RandomForestRegressor()),
  #('GB', GradientBoostingRegressor()),
  #('MLP',MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)),
  ('ET', ExtraTreesRegressor()),
  #('ADA', AdaBoostRegressor()),
]

pipeline = spot_check(a)

#%%
names, results = [], []
for name, model in pipeline:
  cv = KFold(n_splits=10)
  cv_results = cross_val_score(model, x_train, y_train.ravel(), cv=cv, scoring='r2')

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}")

#%%
fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names, rotation='vertical');

#%%
from sklearn.metrics import make_scorer, r2_score
from hyperopt import fmin, tpe, hp

space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    #'max_depth': hp.choice('max_depth', range(1, 20)),
}

scoring = make_scorer(r2_score)
def objective(params):
  cv = KFold(n_splits=10)
  model = ExtraTreesRegressor(**params)
  score = -cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring).mean()
  return score

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
print(best)

#%%
#model = ExtraTreesRegressor(n_estimators=411, max_depth=18)
model = ExtraTreesRegressor(n_estimators=85)
model.fit(x_train, y_train)
pred = model.predict(x_test)
#%%

r2_score(np.exp(y_test), np.exp(pred))