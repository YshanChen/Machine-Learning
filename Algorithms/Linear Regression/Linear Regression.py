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
import matplotlib.pyplot as plt
import seaborn as sns

class LR(object):

    def __init__(self):
        self.W_opt = None
        self.J_function_df = None
        self.X_columns = None

    def _caculate_cost_function(self, y, y_hat, W, lambda_l1, lambda_l2):
        m = y.shape[0]
        J = - 1 / m * ((np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat))).sum() + lambda_l1*(np.abs(W).sum()) + 1/2*lambda_l2*(np.square(W).sum()))
        return J

     # p = pd.DataFrame({'y': pd.Series(map(lambda x: x[0], y)), 'y_hat': pd.Series(map(lambda x: x[0], y_hat)),
     #                  '1-y': pd.Series(map(lambda x: x[0], 1 - y)),
     #                  '1-y_hat': pd.Series(map(lambda x: x[0], 1 - y_hat)),
     #                  'log(1-y_hat)': pd.Series(map(lambda x: x[0], np.log(1 - y_hat))),
     #                  'a': pd.Series(map(lambda x: x[0], np.multiply(y, np.log(y_hat)))),
     #                  'b': pd.Series(map(lambda x: x[0], np.multiply((1 - y), np.log(1 - y_hat)))), 'c': pd.Series(
     #        map(lambda x: x[0], (np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat)))))})

    def _caculate_cost_function_gredient(self, X, y, y_hat, W, w_index, lambda_l1, lambda_l2):
        m = y.shape[0]
        gredient_cost = 1/m * np.multiply((y_hat - y), X.T[w_index].reshape((-1, 1))).sum()
        gredient_l1 = 1/m * np.where(W[w_index] >= 0, 1*lambda_l1, -1*lambda_l1)   # |w| 求导
        gredient_l2 = 1/m * W[w_index]*lambda_l2
        gredient = gredient_cost + gredient_l1 + gredient_l2
        # print((gredient_cost, gredient_l1, gredient_l2, gredient))

        return gredient

    def fit(self, X, y, eta=0.1, iter_rounds=2000, lambda_l1 = 0, lambda_l2 = 0, X_test=None, y_test=None):  # X=X_Train; y=y_Train;lambda_l1 = 0;lambda_l2 = 0.5
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

        for iter in np.arange(0, iter_rounds):  # iter=5
            print('iter:', iter)
            W = W_new

            h_current = self._Logistic_Regression(X=X, W=W)
            # h_current = _Logistic_Regression(self=[],X=X, W=W)
            J = self._caculate_cost_function(y=y, y_hat=h_current, W=W, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
            # J = _caculate_cost_function(self=[], y=y, y_hat=h_current, W=W, lambda_l1=lambda_l1, lambda_l2=lambda_l2)

            # print("J:", round(J, 4))
            J_list[iter] = J
            W_list[iter] = W

            # Test
            if have_test_flag == 1:
                h_current_test = self._Logistic_Regression(X=X_test, W=W)
                J_test = self._caculate_cost_function(y=y_test, y_hat=h_current_test, W=W, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
                J_list_test[iter] = J_test

            W_new = np.zeros((n, 1))
            for w_index in np.arange(0, W.shape[0]):  # w_index=0
                # print(w_index)
                gredient = self._caculate_cost_function_gredient(X=X, y=y, y_hat=h_current, W=W, w_index=w_index, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
                # gredient = _caculate_cost_function_gredient(self=[], X=X, y=y, y_hat=h_current, W=W, w_index=w_index, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
                W_new[w_index] = W[w_index] - eta * gredient

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
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
train_test = pd.concat([train, test], axis=0)

train_test = train_test[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Survived']]
train_test = train_test.fillna(value={'Age': train_test['Age'].mean(), 'Fare':train_test['Fare'].mean()})

### 梯度下降算法 需要 feature scaling
from sklearn import preprocessing
train_test['Age'] = preprocessing.scale(train_test['Age'])
train_test['Fare'] = preprocessing.scale(train_test['Fare'])

train_test, cates = one_hot_encoder(data=train_test,
                               categorical_features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
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
clf.fit(X=X_Train, y=y_Train, eta=0.001, iter_rounds=200000, lambda_l1=0, lambda_l2=0)
functions_df = clf.J_function_df
clf.plot_CostFunction() # 经过 Feature Scaling 后，学习曲线更加平滑下降

pred_Y = clf.predict(new_data=X_test)
pred_dt = pd.DataFrame(y_test)
pred_dt['pred_Y'] = pred_Y
roc_auc_score(pred_dt.Survived, pred_dt.pred_Y)  # 0.8474

# Submit
pre_Y = clf.predict(new_data=test)  # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame({'PassengerId': np.arange(892, 1310), 'Survived': (pre_Y>=0.5).astype('int')})
submit.loc[:, 'Survived'] = submit.loc[:, 'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20190126_LR_l20_FeatureScaling.csv', index=False)

# sklearning
# ------------------- LogisticRegression -----------------
from sklearn.linear_model import
clf = LogisticRegression(solver='liblinear', max_iter=100, verbose=1, n_jobs=-1)
clf.fit(X=X_Train, y=y_Train)
clf.get_params()

pred_dt = pd.DataFrame(clf.predict_proba(X=X_test), index=X_test.index, columns=['pred_0', 'pred_1'])
pred_dt['y_test'] = y_test
roc_auc_score(pred_dt.y_test, pred_dt.pred_1)  # sag=0.8493 liblinear=0.8494

pre_Y = clf.predict_proba(X=test)  # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame(clf.predict_proba(X=test), index=test.index, columns=['pred_0', 'Survived']).drop(columns=['pred_0'], axis=0)
submit['PassengerId'] = np.arange(892, 1310)
submit['Survived'] = (submit['Survived']>=0.5).astype('int')
submit = submit[['PassengerId', 'Survived']]
submit.loc[:, 'Survived'] = submit.loc[:, 'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20190126_LR_sklearn_liblinear.csv', index=False)

# ------------------- SGD -----------------
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', alpha=0.5)
clf.fit(X=X_Train, y=y_Train)
clf.get_params()

pred_dt = pd.DataFrame(clf.predict_proba(X=X_test), index=X_test.index, columns=['pred_0', 'pred_1'])
pred_dt['y_test'] = y_test
roc_auc_score(pred_dt.y_test, pred_dt.pred_1)  # alpha=0.0001:0.7688   alpha=0.5:0.8317

pre_Y = clf.predict_proba(X=test)  # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame(clf.predict_proba(X=test), index=test.index, columns=['pred_0', 'Survived']).drop(columns=['pred_0'], axis=0)
submit['PassengerId'] = np.arange(892, 1310)
submit['Survived'] = (submit['Survived']>=0.5).astype('int')
submit = submit[['PassengerId', 'Survived']]
submit.loc[:, 'Survived'] = submit.loc[:, 'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20190126_LR_sklearn_SGD_l20.5.csv', index=False)


