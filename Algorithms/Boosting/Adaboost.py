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
        self.classifier_list = None
        self.classifier_weights = None
        self.classifier_list_iterproc = None
        self.classifier_weights_iterproc = None

    def _split_feature_point(self, X, y, W):
        E_m_Min = 1
        for feature in X.columns:  # feature = 'x_2'
            feature_values_series = X[feature].sort_values().drop_duplicates(keep='first')  # 排序、去重

            for feature_value_1, feature_value_2 in zip(feature_values_series[0:], feature_values_series[1:]):
                # print(feature_value_1, feature_value_2)  # feature_value_1 = 0.000000; feature_value_2 = 0.000064
                feature_value = round((feature_value_1 + feature_value_2) / 2, 8)  # 中位数作为候选划分数

                # (1,-1) & (-1,1)
                for y_bigger_pred, y_lesser_pred in [(1, -1), (-1, 1)]:
                    y_pred = np.where(X[feature] >= feature_value, y_bigger_pred, y_lesser_pred)

                    # 分类误差率-二分类问题
                    I = (y_pred != y).astype('int')
                    E_m = np.dot(W, I)
                    # dic[(feature, feature_value)] = E_m

                    if E_m < E_m_Min:
                        E_m_Min = E_m
                        feature_opt = feature
                        feature_value_opt = feature_value
                        y_bigger_pred_opt = y_bigger_pred
                        y_lesser_pred_opt = y_lesser_pred
                        y_pred_Min = y_pred

        if E_m_Min >= 0.5:
            print("Error: Base_Classifier error must lesser than 0.5 !")

        return (feature_opt, feature_value_opt, y_bigger_pred_opt, y_lesser_pred_opt, E_m_Min, y_pred_Min)

    def fit(self, X, y): # X = X_train; y = y_train
        # 保存原始X, y
        y[y == 0] = -1  # 0->-1
        X_init = X
        y_init = y

        # 初始化训练集，样本权重, 基学习器权重
        y_class_num = len(y_init.unique())
        data = pd.concat([X_init, y_init], axis=1).rename(str, columns={y_init.name: 'label'})
        W_dic = {1: pd.Series(np.repeat(1/X_init.shape[0], X_init.shape[0]))} # 样本权重-字典
        E_dic = {}  # 误差-字典
        A_dic = {} # 基学习器权重-字典
        classifier_dic = {} # 基学习器-字典

        for iter in np.arange(1, self.params['iter_num']+1): # iter = 1
            print(iter)
            W = W_dic[iter] # m轮样本权重
            X = data.drop(['label'], axis=1)
            y = data['label'].values

            # 对加权的训练集进行模型训练，寻找最小分类误差率对应的分裂特征与分裂点。（因为决策树桩，遍历每个特征计算分类误差率即可）
            base_classifier = self._split_feature_point(X=X, y=y, W=W) # X=X; y=y; W=W
            # print("m 轮样本权重的唯一值: {}".format(W.unique()))
            print("第 {} 轮， 最优特征：{}, 最优切分点: {}, 最小误差率: {}".format(iter, base_classifier[0], base_classifier[1], base_classifier[4]))
            # base_classifier = (feature_opt, feature_value_opt, y_bigger_pred_opt, y_lesser_pred_opt, E_m_Min, y_pred_Min)
            if base_classifier[4] >= 0.5:  # 如果E_m >= 0.5 停止迭代，跳出循环;
                print("Error >= 0.5, Boosting stopped. ")
                print(iter, base_classifier[0], base_classifier[1], base_classifier[4])
                break
            classifier_dic[iter] = base_classifier

            # 基学习器分类误差率(二分类问题)
            E_m = classifier_dic[iter][4]
            E_dic[iter] = E_m

            # 基学习器的权重
            A_m = 0.5*np.log((1-E_m)/E_m)
            A_dic[iter] = A_m

            # 更新样本权重分布 m+1轮样本权重
            y_pred = classifier_dic[iter][5]
            Z_m = np.dot(W, np.exp(-A_m*y*y_pred))
            W_dic[iter+1] = W*np.exp(-A_m*y*y_pred)/Z_m

            # 计算截止到此轮，学习器G的分类误差率
            self.classifier_list_iterproc = classifier_dic
            self.classifier_weights_iterproc = A_dic

            y_pred_iterproc = self._predict_iterproc(X)
            I_iterproc = (y_pred_iterproc != y).astype('int')
            E_iterproc = I_iterproc.sum()/I_iterproc.count()
            print("第 {} 轮 训练误差： {}".format(iter, round(E_iterproc, 4)))

        # 输出基学习器序列和基学习器权重
        self.classifier_list = classifier_dic
        self.classifier_weights = A_dic

    def _predict_iterproc(self, new_data): # new_data = X_test
        # boosting 基学习器个数
        classifiers_num = len(self.classifier_list_iterproc)

        predict_Y = pd.Series([])
        # 逐条样本预测
        for row_index, row_data in new_data.iterrows(): # row_index = 495
            # print(row_index)
            # print('----------------------')
            # row_data = new_data.iloc[1]

            # y_pred_classifier_list = pd.Series(np.zeros((7)).tolist())
            y_pred_classifier_list = pd.Series(np.zeros((classifiers_num)).tolist())
            for iter in np.arange(1, classifiers_num+1): # iter = 4
                # print(iter)
                base_classifier = self.classifier_list_iterproc[iter] # base_classifier = classifier_list[iter]
                feature_opt = base_classifier[0]
                feature_value_opt = base_classifier[1]
                y_bigger_pred_opt = base_classifier[2]
                y_lesser_pred_opt = base_classifier[3]
                y_pred_classifier = np.where(row_data[feature_opt] >= feature_value_opt, y_bigger_pred_opt, y_lesser_pred_opt)
                y_pred_classifier_list[iter-1] = y_pred_classifier
            G = np.dot(pd.Series(self.classifier_weights_iterproc), y_pred_classifier_list)
            f = np.where(G>=0, 1, 0) # 原始为（1，0），训练过程中转为了（1，-1）
            predict_Y[row_index] = f

        return predict_Y

    def predict(self, new_data): # new_data = X_test
        # boosting 基学习器个数
        classifiers_num = len(self.classifier_list)

        predict_Y = pd.Series([])
        # 逐条样本预测
        for row_index, row_data in new_data.iterrows(): # row_index = 495
            # print('----------------------')
            # print("第 {} 个样本".format(row_index))
            # row_data = new_data.iloc[1]

            # y_pred_classifier_list = pd.Series(np.zeros((7)).tolist())
            y_pred_classifier_list = pd.Series(np.zeros((classifiers_num)).tolist())
            for iter in np.arange(1, classifiers_num+1): # iter = 4
                # print("第 {} 轮".format(iter))
                base_classifier = self.classifier_list[iter] # base_classifier = classifier_list[iter]
                feature_opt = base_classifier[0]
                feature_value_opt = base_classifier[1]
                y_bigger_pred_opt = base_classifier[2]
                y_lesser_pred_opt = base_classifier[3]
                y_pred_classifier = np.where(row_data[feature_opt] >= feature_value_opt, y_bigger_pred_opt, y_lesser_pred_opt)
                y_pred_classifier_list[iter-1] = y_pred_classifier
                # print("第 {} 轮预测 {}".format(iter, y_pred_classifier))
            G = np.dot(pd.Series(self.classifier_weights), y_pred_classifier_list)
            f = np.where(G>=0, 1, 0) # 原始为（1，0），训练过程中转为了（1，-1）
            print("总学习器预测: weights: {}, G: {}, pred: {}".format(self.classifier_weights, round(G, 4), f))
            predict_Y[row_index] = f

        return predict_Y




# Kaggle Tanic --------------------------------------------------------------------------------
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

# test
cls = Adaboost(iter_num=200)
cls.fit(X_train, y_train)
y_test_pred = cls.predict(X_test)

print('AUC for Test : ', roc_auc_score(y_test, y_test_pred)) # AUC for Test : 0.7678(iter=100)
print('Error for Test : ', (y_test != y_test_pred).sum()/len(y_test))  # Error for Test :  0.2156(iter=100)

# 第一棵树的误差 0.7576 0.2213

# Submit
pre_Y = cls.predict(new_data=test)   # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
submit.to_csv('Result/submit_20181201(Adaboost).csv', index=False)

# iris data --------------------------------------------------------------
from sklearn.datasets import load_iris
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]
def score(y, pred):
    I = (y==pred).astype('int32')
    score = I.sum()/I.count()
    print(score)

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = pd.DataFrame(X_train, columns=['x1', 'x2'])
y_train = pd.Series(y_train, name='y')
X_test = pd.DataFrame(X_test, columns=['x1', 'x2'])
y_test = pd.Series(y_test, name='y')

# iter = 10
clf = Adaboost(iter_num=10)
clf.fit(X_train, y_train)
pred = clf.predict(new_data=X_test)
df_predict = pd.DataFrame({'y':y_test.map({1:1, -1:0}), 'pred':pred})
score(y=df_predict['y'], pred=df_predict['pred']) # 0.9696969696969697

# iter = 100
clf = Adaboost(iter_num=100)
clf.fit(X_train, y_train)
pred = clf.predict(new_data=X_test)
df_predict = pd.DataFrame({'y':y_test.map({1:1, -1:0}), 'pred':pred})
score(y=df_predict['y'], pred=df_predict['pred']) # 0.8787878787878788

# sklearn
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  # 0.9393939393939394