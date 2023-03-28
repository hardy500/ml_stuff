#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

from sklearn.model_selection import KFold, cross_val_score, RepeatedStratifiedKFold
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
  for i in a:
    c.append(
      (i[0], Pipeline([
      (i[0], StandardScaler()),
      (i[2][0], i[2][1])
     ]))
    )
  return c


#%%
strings = ['PassengerId','HomePlanet', 'CryoSleep', 'Cabin', 'Destination','VIP']
numbers = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

imputer = KNNImputer()
scaler = StandardScaler()
x1 = encode(x[strings], strings)
x2 = pd.DataFrame(imputer.fit_transform(x[numbers]), columns=numbers)
#x2 = pd.DataFrame(scaler.fit_transform(x2), columns=numbers)

#%%
lof = LocalOutlierFactor()
outliers = lof.fit_predict(x2)
# Select all rows that are not outlier
mask = outliers != -1
x = pd.concat([x2, x1], axis=1)[mask]
lb = LabelBinarizer()
y = lb.fit_transform(y)[mask]

#%%
features = [8701, 1, 4, 5, 8698, 15264, 0, 2, 15266, 15218, 9273, 8893, 8699]
x = x.iloc[:,features]

#%%
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=.1, shuffle=True, random_state=42, stratify=y
)

#%%

#fs = SelectKBest(score_func=f_classif, k='all')
#fs.fit(x_train, y_train.ravel())
#x_train_fs = fs.transform(x_train)
#x_test_fs = fs.transform(x_val)



# what are the scores for the features
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



#%%
a = [
  ('ScaledLR', 'Scalar', ('LR', LogisticRegression())),
  #('ScaledLDA', 'Scalar', ('LDA', LinearDiscriminantAnalysis())),
  #('ScaledKNN', 'Scalar', ('KNN', KNeighborsClassifier())),
  #('ScaledCART', 'Scalar', ('CART', DecisionTreeClassifier())),
  ('ScaledSVC', 'Scalar', ('SVC', SVC())),
  #('ScaledSGD', 'Scalar', ('SGD', SGDClassifier())),
  #('ScaledNB', 'Scalar', ('NB', GaussianNB())),

  ('ScaledAB', 'Scalar', ('AB', AdaBoostClassifier())),
  ('ScaledGDM', 'Scalar', ('GDM', GradientBoostingClassifier())),
  ('ScaledHGB', 'Scalar', ('HGB', HistGradientBoostingClassifier())),
  ('ScaledRF', 'Scalar', ('RF', RandomForestClassifier())),
  ('ScaledXG', 'Scalar', ('XG', XGBClassifier())),
  ('ScaledET', 'Scalar', ('ET', ExtraTreesClassifier())),
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