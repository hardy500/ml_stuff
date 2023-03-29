#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import (
  AdaBoostClassifier, GradientBoostingClassifier,
  HistGradientBoostingClassifier,
  RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

#%%
y = train_data.pop('Transported')
x = train_data
#%%
x[['Deck', 'Cabin_Num', 'Side']]  = x['Cabin'].str.split('/', expand=True)
x = x.drop('Cabin', axis=1)


test_data[['Deck', 'Cabin_Num', 'Side']]  = test_data['Cabin'].str.split('/', expand=True)
test_data = test_data.drop('Cabin', axis=1)

#%%
def impute(x, col, strategy="most_frequent"):
  if strategy == "most_frequent":
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(x[col]), columns=x[col].columns)
  else:
   imputer = SimpleImputer(strategy=strategy)
   df_imputed = pd.DataFrame(imputer.fit_transform(x[col]), columns=x[col].columns)
  return df_imputed

def encode(x, features):
  encoder = OneHotEncoder(drop='first', sparse=False)
  encoded = encoder.fit_transform(x)
  x = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(features))
  return x

def spot_check(a):
  c = []
  for i in range(len(a)):
    c.append(a[i])
  return c


#%%
strings = ['HomePlanet', 'CryoSleep','Destination','VIP','Deck', 'Side']
numbers = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_Num']

imputer = KNNImputer()
scaler = MinMaxScaler()
x1 = encode(x[strings], strings)
x2 = pd.DataFrame(imputer.fit_transform(x[numbers]), columns=numbers)
x2 = pd.DataFrame(scaler.fit_transform(x2), columns=numbers)

#%%
# TEST DATA
x1_test = encode(test_data[strings], strings)
x2_test = pd.DataFrame(imputer.fit_transform(test_data[numbers]), columns=numbers)
x2_test = pd.DataFrame(scaler.fit_transform(x2_test), columns=numbers)

#%%
lof = LocalOutlierFactor()
outliers = lof.fit_predict(x2)
# Select all rows that are not outlier
mask = outliers != -1
x = pd.concat([x2, x1], axis=1)[mask]
lb = LabelBinarizer()
y = lb.fit_transform(y)[mask]

#%%

#%%
#features = [8701, 1, 4, 5, 8698, 15264, 0, 2, 15266, 15218, 9273, 8893, 8699]
features = [10, 1, 4, 5, 7, 17, 18, 20, 25, 13, 21, 0, 6, 2, 19, 15, 8, 3, 23]
x = x.iloc[:,features]
#%%
# TEST DATA
x_test = pd.concat([x2_test, x1_test], axis=1)
x_test = x_test.reindex(columns=x.columns, fill_value=0)

#x_test = x_test[mask].iloc[:, features]

#%%
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=.1, shuffle=True, random_state=42, stratify=y
)

##%%
#
#fs = SelectKBest(score_func=f_classif, k='all')
#fs.fit(x_train, y_train.ravel())
#x_train_fs = fs.transform(x_train)
#x_test_fs = fs.transform(x_val)
#
#
#
## what are the scores for the features
##%%
#scores = []
#for i in range(len(fs.scores_)):
#  print(f"Feature {i}: {fs.scores_[i]:.3f}")
#  if f"{fs.scores_[i]}" != 'nan':
#    print("NAN")
#    scores.append((fs.scores_[i], i))
#
##%%
#scores = sorted(scores, key=lambda x: x[0], reverse=True)
#scores



#%%
a = [
  #('LR', LogisticRegression()),
  #('LDA', LinearDiscriminantAnalysis()),
  #('KNN', KNeighborsClassifier()),
  #('CART', DecisionTreeClassifier()),
  #('SVC', SVC()),
  #('SGD', SGDClassifier()),
  #('NB', GaussianNB()),

  #('AB', AdaBoostClassifier()),
  #('GDM', GradientBoostingClassifier()),
  ('HGB', HistGradientBoostingClassifier()),
  #('RF', RandomForestClassifier()),
  #('XG', XGBClassifier()),
  #('ET', ExtraTreesClassifier()),
  #('MLP', MLPClassifier(max_iter=1000, random_state=42)),
]


pipelines = spot_check(a)

#%%

names, results = [],[]
n_splits = 10
for name, model in pipelines:
  cv = KFold(n_splits=n_splits)
  cv_results = cross_val_score(model, x_train, y_train.ravel(), cv=cv, scoring='accuracy')

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}")

#%%
# Compare the algorithm
fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names, rotation='vertical');

#%%

from hyperopt import fmin, tpe, hp

# Define the hyperparameter search space
space = {
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'max_depth': hp.choice('max_depth', range(1, 10)),
    #'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 50)),
    #'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
    #'l2_regularization': hp.loguniform('l2_regularization', -10, 0)
}

# Define the objective function to minimize
def objective(params):
    clf = HistGradientBoostingClassifier(**params, max_iter=1000)
    cv = KFold(n_splits=10)
    score = cross_val_score(clf, x_train, y_train.ravel(), cv=cv, scoring='accuracy').mean()
    return -score

best = fmin(objective, space, algo=tpe.suggest, max_evals=150)
print("Best hyperparameters:", best)

#%%

# Best hyperparameters: {'learning_rate': 0.07358629094211126, 'max_depth': 8}
model =  HistGradientBoostingClassifier(learning_rate=0.07358629094211126, max_depth=8)
model.fit(x_train, y_train.ravel())

pred = model.predict(x_test)
#%%

pred = [True if i == 1 else False for i in pred]

data = {
  'PassengerId': test_data['PassengerId'].values,
  'Transported': pred
}

sol = pd.DataFrame(data)
sol.to_csv('summit.csv', index=False)
