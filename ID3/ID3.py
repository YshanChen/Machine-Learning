# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from math import log
from treelib import *
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from sklearn import preprocessing
import re

# ------------------------- 数据集 --------------------------- #
df = pd.read_csv('E:/Machine Learning/Algorithm implementation/Data/ID3_1.csv',encoding="GBK")
df.info()

# 处理数据集
df.age = np.where(df.age == "青年",1,
                  np.where(df.age == "中年",2,3))
df.loan = np.where(df.loan == "一般",1,
                   np.where(df.loan == "好",2,3))

df.work = np.where(df.work == "是",1,0)
df.hourse = np.where(df.hourse == "是",1,0)
df['class'] = np.where(df['class'] == "是",1,0)

for i in np.arange(len(df.columns)):
    df.ix[:,i] = df.ix[:,i].astype('category')


# ----------------------------------  信息增益算法 ------------------------------------- #

# 特征分裂向量
def feature_split(df,y):
    feature_split_num = []

    # Y类别个数
    class_categories_num = len(df[y].cat.categories)

    for i in np.arange(0,(len(df.columns) - 1)):

        # 特征类别个数
        features_categories_num = len(df[df.columns[i]].cat.categories)

        Vec = np.zeros(shape=(features_categories_num,class_categories_num))
        for j in np.arange(0,len(df[df.columns[i]].cat.categories)):
            a = ((df[df.columns[i]] == df[df.columns[i]].cat.categories[j]) & (df[y] == df[y].cat.categories[0])).sum()
            b = ((df[df.columns[i]] == df[df.columns[i]].cat.categories[j]) & (df[y] == df[y].cat.categories[1])).sum()
            Vec[j] = np.array([[a,b]],dtype='float64')

        feature_split_num.append(Vec)

    return feature_split_num


# 数据集D的经验熵
def entropy(Di_vec):
    D = Di_vec.sum()
    if D == 0:
        p_vec = np.zeros(shape=(np.shape(Di_vec)))
    else:
        p_vec = Di_vec / D
    h_vec = np.array([])

    for p in p_vec:
        if p != 0:
            h = p * log(p,2)
            h_vec = np.append(h_vec,h)
        else:
            h_vec = np.append(h_vec,0)
    H = -(h_vec.sum())

    return (H)


# 特征A对数据集D的条件熵
def con_entroy(Di_vec,Aik_vec):
    H_Di = np.array([])
    P_Di = np.array([])
    for D_i in Aik_vec:
        H_Di = np.append(H_Di,entropy(D_i))
        P_Di = np.append(P_Di,(D_i.sum() / Di_vec.sum()))
    H_DA = (H_Di * P_Di).sum()

    return (H_DA)


# 特征A的信息增益
def gain(Di_vec,Aik_vec):
    gain = entropy(Di_vec) - con_entroy(Di_vec,Aik_vec)

    return (gain)


# 计算每个特征的信息增益，并取最大值
def gain_max(df,y):
    gain_vec = np.zeros(shape=((len(df.columns) - 1),1))

    feature_split_num = feature_split(df,y)

    # Y类别个数
    Di_vec = np.array(df[y].value_counts())

    # 计算各特征信息增益
    for i in np.arange(0,len(feature_split_num)):
        gain_vec[i] = gain(Di_vec,feature_split_num[i])

    # 选取信息增益最大的特征
    return [df.columns[gain_vec.argmax()],gain_vec.max()]


# -------------------------------- ID3 算法  -------------------------------------- #
def merge_two_dicts(x,y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


pp = df
df = pp
y = 'class'
delta = 0.005

DTree = {}

max_class_in_D = df[y].value_counts().argmax()  # D中实例最大的类

if gain_max(df,y)[1] >= delta:
    split_feature_name = gain_max(df,y)[0]

    # 初次分裂
    for cat in np.unique(df[split_feature_name]):

        # cat = 1
        df_split_temp = df[df[split_feature_name] == cat].drop(split_feature_name,axis=1)
        description = ' '.join([str(split_feature_name),'=',str(cat)])

        currentValue = df_split_temp

        if gain_max(df_split_temp,y)[1] < delta:
            currentValue = max_class_in_D

        if (len(df_split_temp[y].unique()) == 1):
            currentValue = df[y].values[0]

        if df_split_temp.empty == True:
            currentValue = max_class_in_D

        currentTree = {description: currentValue}
        DTree.update(currentTree)


def Decision_Tree(DTree,y='class',delta=0.005):
    for key,value in DTree.items():
        print([key,value])
        print('-----------------------------')

        subTree = {}

        # key = 'car = 0'
        # value = DTree[key]

        if isinstance(value,pd.DataFrame):
            df = value

            split_feature_name = gain_max(df,y)[0]

            for cat in np.unique(df[split_feature_name]):

                # cat = 1
                df_split_temp = df[df[split_feature_name] == cat].drop(split_feature_name,axis=1)
                description = ' '.join([str(split_feature_name),'=',str(cat)])

                if (gain_max(df_split_temp,y)[1] >= delta) and (len(df_split_temp[y].unique()) != 1) and (
                        df_split_temp.empty != True):

                    currentTree = {description: df_split_temp}
                    currentValue = Decision_Tree(currentTree,y='class',delta=0.005)

                    subTree.update(currentValue)

                else:

                    if gain_max(df_split_temp,y)[1] < delta:
                        currentValue = max_class_in_D

                    if (len(df_split_temp[y].unique()) == 1):
                        currentValue = df_split_temp[y].values[0]

                    if (df_split_temp.empty == True):
                        currentValue = max_class_in_D

                    subTree.update({description: currentValue})

            DTree[key] = subTree

        else:
            print(99)

    return DTree


q = Decision_Tree(DTree,y='class',delta=0.005)

# --------------------------- 预测函数 ----------------------------- #
DicTree = {'car=0': {'age=2': 1,'age=3': 1,'age=1': {'loan=1': 0,'loan=2': 1}},
           'car=1': {'money=1': 0,'money=4': 0,'money=3': {'age=1': 1,'age=2': {'hourse=0': 0,'hourse=1': 1}},
                     'money=2': {'hourse=1': 0,'hourse=0': 1}}}


def ID3_predict_one(DTree,row_data):
    for keys,values in DTree.items():
        T_key = keys
        T_value = values

        T_key_list = re.split('(=|<|<=|>|>=|!=)',T_key)
        split_feature = T_key_list[0].strip()
        split_feature_oper = T_key_list[1].strip()
        split_feature_value = T_key_list[2].strip()

        if str(row_data[split_feature]) == split_feature_value:
            if isinstance(T_value,dict):
                return ID3_predict_one(T_value,row_data)
            else:
                return T_value


def ID3_predict(DTree,new_data):
    predict_Y = []

    for row_data in new_data.iterrows():
        # row_data_series = temp[0]

        row_data_series = row_data[1]
        predict_Y.append(ID3_predict_one(DTree,row_data_series))

    return (predict_Y)


predict_Y = ID3_predict(DTree=q,new_data=df)







