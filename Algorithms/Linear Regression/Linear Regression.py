# -*- coding: utf8 -*-
"""
Created on 2019/01/17
@author: Yshan.Chen

Linear Regression
Gradient Descend

"""

"""
Todo list:
1. 加入正则项 L1， L2
"""

import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class LR(object):

    def __init__(self):
        self.W_opt = None
        self.J_function_df = None
        self.X_columns = None

    def fit(self, X, y, eta=0.1, iter_rounds=2000, regularization=None, X_test=None, y_test=None):  # X=X_Train; y=y_Train
        if X_test is not None and y_test is not None:
            have_test_flag = 1
            X_test = X_test.values
            y_test = y_test.reshape((-1, 1))
            J_list_test = {}
        else:
            have_test_flag = 0

        self.X_columns = X.columns.values
        Data = X.copy()
        Data['label'] = y
        m = Data.shape[0]  # sample number
        n = X.shape[1]  # features number

        X = Data.drop(['label'], axis=1).values
        y = Data['label'].values.reshape((-1, 1))
        W_new = np.random.rand(n, 1) * 0.0001 # 初始化参数

        J_list = {}
        W_list = {}

        for iter in np.arange(0, iter_rounds):  # iter=34
            print('iter:', iter)
            W = W_new

            h_current = self._Logistic_Regression(X=X, W=W)
            J = - 1 / m * (np.multiply(y, np.log(h_current)) + np.multiply((1 - y), np.log(1 - h_current))).sum()

            # print("J:", round(J, 4))
            J_list[iter] = J
            W_list[iter] = W

            # Test
            if have_test_flag == 1:
                h_current_test = self._Logistic_Regression(X=X_test, W=W)
                J_test = - 1 / m * (np.multiply(y_test, np.log(h_current_test)) + np.multiply((1 - y_test), np.log(
                    1 - h_current_test))).sum()
                J_list_test[iter] = J_test

            W_new = np.zeros((n, 1))
            for x_index in np.arange(0, W.shape[0]):  # x_index=0
                # print(x_index)
                W_new[x_index] = W[x_index] - eta * (np.multiply((h_current - y), X.T[x_index].reshape((-1, 1))).sum() / m)

        # 学习曲线
        J_function_df = pd.DataFrame(list(J_list.items()), columns=['iter', 'J_function'])
        if have_test_flag == 1:
            J_function_df['J_function_Test'] = J_list_test.values()
        self.J_function_df = J_function_df

        # 最优参数
        self.W_opt = W_list[iter_rounds - 1]

    def predict(self, new_data):  # 逐条预测，未实现并行化
        if self.W_opt is None:
            print("There is no optimal parameters, fit a model first !")
        elif (self.X_columns == new_data.columns.values).all():
            pred_Y = self._Logistic_Regression(X=new_data, W=self.W_opt)
            pred_Y = pred_Y.reshape((-1, ))
        else:
            print("New Data columns is not same as Train !")
        return pred_Y

    def plot_CostFunction(self):
        if self.J_function_df is None:
            print("There is no cost function Dataframe, fit a model first !")
        else:
            columns = self.J_function_df.columns[1:]
            mycolors = ['tab:red', 'tab:blue']

            plt.figure(figsize=(16, 10), dpi=80)
            for i, column in enumerate(columns):
                plt.plot('iter', column, data=self.J_function_df, color=mycolors[i])
                plt.text(self.J_function_df.shape[0] + 20, self.J_function_df[column].values[-1], column, fontsize=14, color=mycolors[i])

            plt.title("Cost function", fontsize=22)
            plt.xlabel("Iter")
            plt.ylabel("Cost Function")
            plt.grid(axis='both', alpha=.3)
            plt.show()

    def _Logistic_Regression(self, X, W):
        # X shape is (sample_num, features_num)
        # W shape is (features_num, 1)
        # h shape is (sample_num, 1)

        if X.shape[1] != W.shape[0]:
            print("Error: (dim 1) != (dim 0) !")
        else:
            sigma = np.dot(X, W)
            h = 1 / (1 + np.exp(-sigma))
        return h

# ------------------------------- Test ----------------------------------
# 2.Kaggle Titanic Data [binary]
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# 读取数据
train = pd.read_csv('Data/train_fixed.csv')
test = pd.read_csv('Data/test_fixed.csv')
train_test = pd.concat([train, test], axis=0)

train_test, cates = one_hot_encoder(data=train_test,
                               categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                               nan_as_category=False)

train = train_test[~np.isnan(train_test['Survived'])]
test = train_test[np.isnan(train_test['Survived'])].drop(['Survived'], axis=1)

# 分割数据
train_train, train_test = train_test_split(train, test_size=0.4, random_state=0)
X_train = train_train.drop(['Survived'], axis=1)
y_train = train_train['Survived']
X_test = train_test.drop(['Survived'], axis=1)
y_test = train_test['Survived']
X_Train = train.drop(['Survived'], axis=1)
y_Train = train['Survived']

# Test
clf = LR()
clf.fit(X=X_Train, y=y_Train, eta=0.03, iter_rounds=5000,  X_test=X_test, y_test=y_test)
clf.J_function_df
clf.plot_CostFunction()

pred_Y = clf.predict(new_data=X_test)
pred_dt = pd.DataFrame(y_test)
pred_dt['pred_Y'] = pred_Y
roc_auc_score(pred_dt.Survived, pred_dt.pred_Y)  # 0.8578

# Submit
pre_Y = clf.predict(new_data=test)  # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame({'PassengerId': np.arange(892, 1310), 'Survived': (pre_Y>=0.5).astype('int')})
submit.loc[:, 'Survived'] = submit.loc[:, 'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20190119_LR.csv', index=False)
