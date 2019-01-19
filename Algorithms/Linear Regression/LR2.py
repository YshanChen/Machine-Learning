# -*- coding: utf8 -*-
"""
Created on 2019/01/17
@author: Yshan.Chen

Linear Regression
Gradient Descend

"""

"""

"""

import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time
import sys

def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# 读取数据
train = pd.read_csv('Data/train_fixed.csv')
test = pd.read_csv('Data/test_fixed.csv')

train, cates = one_hot_encoder(data=train,
                               categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                               nan_as_category=False)
test, cates = one_hot_encoder(data=test,
                              categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                              nan_as_category=False)

# 分割数据
train_train, train_test = train_test_split(train, test_size=0.4, random_state=0)
X_train = train_train.drop(['Survived'], axis=1)
y_train = train_train['Survived']
X_test = train_test.drop(['Survived'], axis=1)
y_test = train_test['Survived']
X_Train = train.drop(['Survived'], axis=1)
y_Train = train['Survived']

# fit

def Sigmoid(X, W): 
    # X shape is (sample_num, features_num)
    # W shape is (features_num, 1)
    # h shape is (sample_num, 1)

    if X.shape[1] != W.shape[0]:
        print("Error: (dim 1) != (dim 0) !")
    else:
        sigma = np.dot(X, W)
        h = 1/(1+np.exp(-sigma))
    return h

def fit(X, y, regularization = 'None'): # X=X_Train; y=y_Train
    Data = X.copy()
    Data['label'] = y
    m = Data.shape[0] # sample number
    n = X.shape[1] # features number
    X_columns = X.columns

    X = Data.drop(['label'], axis=1).values
    y = Data['label'].values
    W = np.random.rand(n, 1)

    # 固定学习率 eta
    eta = 0.1

    J_list = {}
    W_list = {}
    for iter in np.arange(0, 30): # iter=0
        print('iter:', iter)
        h_current = Sigmoid(X, W)
        J = - 1/m * (np.multiply(y, np.log(h_current)) + np.multiply((1 - y), np.log(1 - h_current))).sum()
        J_list[iter] = J
        W_list[iter] = W

        for x_index in np.arange(0, W.shape[0]): # x_index=0
            W[x_index] = W[x_index] - eta*(np.multiply((h_current-y), X.T[iter]).sum()/m)


