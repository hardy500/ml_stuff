#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('data/Iris.csv')
x = df[df.columns.values[1:-1]]
y = df['Species']

#%%
le = LabelEncoder()
y = le.fit_transform(y)

#%%
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

def spot_check(a):
  c = []
  for i in range(len(a)):
    c.append(a[i])
  return c

a = [
  ('LR', LogisticRegression()),
  ('LDA', LinearDiscriminantAnalysis()),
  ('KNN', KNeighborsClassifier()),
  ('CART', DecisionTreeClassifier()),
  ('SVC', SVC()),
  ('SGD', SGDClassifier()),
  ('NB', GaussianNB()),

  ('AB', AdaBoostClassifier()),
  ('GDM', GradientBoostingClassifier()),
  ('HGB', HistGradientBoostingClassifier()),
  ('RF', RandomForestClassifier()),
  ('XG', XGBClassifier()),
  ('ET', ExtraTreesClassifier()),
  #('MLP', MLPClassifier(max_iter=1000, random_state=42)),
]

pipelines = spot_check(a)

#%%
from sklearn.model_selection import KFold, cross_val_score

names, results = [], []
for name, model in pipelines:
  cv = KFold(n_splits=10)
  cv_result = cross_val_score(model, x_train, y_train.ravel(), cv=cv, scoring='accuracy')

  results.append(cv_result)
  names.append(name)
  print(f"{name}: {cv_result.mean():.3f}")

#%%
fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names, rotation='vertical');
LinearDiscriminantAnalysis
#%%
from sklearn.metrics import accuracy_score

# Final Model
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")