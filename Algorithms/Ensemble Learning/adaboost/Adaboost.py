# -*- coding: utf8 -*-
"""
Created on 2018/11/04
@author: Yshan.Chen

Update: 2018/11/04

Commit：
实现Adaboost算法, 基学习器为决策树桩。
1.

Todo List:
1.
"""

import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time
import sys

# from Algorithms.DecisionTree.CART import CART

class Adaboost(object):
    """
    1. 基学习器为决策树桩
    2. Adaboost只支持二分类
    """

    def __init__(self, classifier='CART', iter_num=20, early_stopping_num=1e10):
        self.params = {'classifier':classifier,
                       'iter_num':iter_num,
                       'early_stopping_num':early_stopping_num}

    def fit(self, X, y): # X = X; y = y
        # 保存原始X, y
        y[y==0] = -1 # 0->-1
        X_init = X
        y_init = y

        # 初始化训练集，样本权重, 基学习器权重
        data = pd.concat([X_init, y_init], axis=1).rename(str, columns={y_init.name: 'label'})
        data['label'] = data['label'].astype('category')
        W_dic = {1: pd.Series(np.repeat(1/X_init.shape[0], X_init.shape[0]))} # 样本权重-字典
        A_dic = {}
        E_dic = {}

        for iter in np.arange(1, self.params['iter_num']+1): # iter = 2
            W = W_dic[iter] # m轮样本权重
            X = data.drop(['label'], axis=1)
            y = data['label']

            # 对加权的训练集进行模型训练，寻找最小分类误差率。（因为决策树桩，遍历每个特征计算分类误差率即可）


            # 分类误差率-二分类问题
            I = (y_pred != y).astype('int')
            E_m = np.dot(W, I)
            E_dic[iter] = E_m

            # 基学习器的权重
            A_m = 0.5*np.log((1-E_m)/E_m)
            A_dic[iter] = A_m

            # 更新样本权重分布 m+1轮样本权重
            Z_m = np.dot(W, np.exp(-A_m*y*y_pred))
            W_dic[iter+1] = W*np.exp(-A_m*y*y_pred)/Z_m

    return A_dic, G_dic


train = pd.read_csv('Data/train_fixed.csv')
test = pd.read_csv('Data/test_fixed.csv')

# onehot
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns
train, cates = one_hot_encoder(data=train,
                              categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                              nan_as_category=False)
test, cates = one_hot_encoder(data=test,
                              categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                              nan_as_category=False)

# 分割数据
train_train, train_test = train_test_split(train,test_size=0.4,random_state=0)

X_train = train_train.drop(['Survived'], axis=1)
y_train = train_train['Survived']
X_test = train_test.drop(['Survived'], axis=1)
y_test = train_test['Survived']
X = train.drop(['Survived'], axis=1)
y = train['Survived']
