# -*- coding: utf8 -*-
"""
Created on 2019/01/07
@author: Yshan.Chen

原始的GBDT算法，考虑带正则项。基学习器为决策树，应用自己编写的CART算法。

"""

"""
问题：
1. P的线性搜索区间？
"""


sys.path.append('Algorithms/DecisionTree')
from CART import CART
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time
import sys

# 回归
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

train = pd.read_csv('data/boston_train.csv')
test = pd.read_csv('data/boston_test.csv')
submission = pd.read_csv('data/boston_submisson_example.csv')
train_X = train.drop(['ID', 'medv'], axis=1)
train_Y = train['medv']
train_X, cates = one_hot_encoder(data=train_X, categorical_features=['rad'], nan_as_category=False)
test_X = test.drop(['ID'], axis=1)
test_X, cates = one_hot_encoder(data=test_X, categorical_features=['rad'], nan_as_category=False)


m = train_X.shape[0]
K = 10
loss_function = 'square loss'

F0 = 1/m*train_Y.mean()

for k in np.arange(1,K+1):  # k = 1
    print("第", k, "个基学习器：")

    # 0. 设定 F_pre_k
    if k == 1:
        F_pre_k = F0
    else:
        F_pre_k = F_k

    # 1. 计算 response, 负梯度；
    if loss_function == 'square loss':
        y_reponse = (train_Y - F_pre_k).rename("y_response")

    # 2. 构建一个基学习器，拟合y_reponse. 即针对数据集{x,y}=>{x,y_reponse}构建学习器；
    f_k_learner = CART(objective='regression', max_depth=5)
    f_k_learner.fit(X=train_X, y=y_reponse)

    # 3. 线性搜索 p, 最小化损失函数. 这里比对的是真实值y
    f_k = f_k_learner.predict(new_data=train_X)  # 第k次的预测值，预测的reponse/负梯度/残差

    loss_min = 1e8
    for p in np.arange(0.5,5,0.05): # p = 1
        F_k_temp = F_pre_k + p*f_k
        loss = np.dot((train_Y - F_k_temp), (train_Y - F_k_temp))
        # print(loss)
        if loss <= loss_min:
            loss_min = loss
            p_opt = p
    print("最小损失： ", loss_min)

    # 4. 赋值： fk = p*fk; Fk = Fk-1 + fk
    f_k = p_opt * f_k
    F_k = F_pre_k + f_k





































# 3. Boston Housing [regression] ------------------------------------------
# onehot
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

train = pd.read_csv('data/boston_train.csv')
test = pd.read_csv('data/boston_test.csv')
submission = pd.read_csv('data/boston_submisson_example.csv')
train_X = train.drop(['ID', 'medv'], axis=1)
train_Y = train['medv']
train_X, cates = one_hot_encoder(data=train_X, categorical_features=['rad'], nan_as_category=False)
test_X = test.drop(['ID'], axis=1)
test_X, cates = one_hot_encoder(data=test_X, categorical_features=['rad'], nan_as_category=False)

rgs = CART(objective='regression', max_depth=5)
rgs.params
rgs.fit(X=train_X, y=train_Y)
rgs.DTree
test_y_pred = rgs.predict(new_data=test_X)
