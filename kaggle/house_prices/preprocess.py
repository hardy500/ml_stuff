#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_xy(data):
  x = data.drop('Id', axis=1)
  y = data.pop('SalePrice')
  return x, y

def encode(x):
  has_nan = [idx for idx, value in enumerate(x.isna().sum().values) if value != 0]
  numbers = x.iloc[:,has_nan].select_dtypes(include=['float'])
  strings = x.iloc[:,has_nan].select_dtypes(include=['object'])

  x_num = numbers.apply(lambda x: x.fillna(x.median()))
  x_strings = pd.get_dummies(strings, drop_first=True)

  return x_num, x_strings

def get_data(train=True):
  if train:
    train_data = pd.read_csv('datasets/train.csv')
    x, y = get_xy(train_data)
  else:
    train_data = pd.read_csv('datasets/test.csv')
    idd = train_data.pop('Id')
    x = train_data

  x1, x2 = encode(x)
  scaler = StandardScaler()
  x1 = pd.DataFrame(scaler.fit_transform(x1), columns=x1.columns)
  x = pd.concat([x1, x2], axis=1)

  if train == False:
    return x, idd

  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
  return x_train, x_val, y_train, y_val