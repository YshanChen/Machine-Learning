# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time

# 信息增益算法 -----------------------------------------------------
def order_Y(data,y):
    df = data.copy()
    df['label'] = df[y]
    df = df.drop([y],axis=1)
    return df

# 特征分裂向量
def feature_split(df,y):
    feature_split_num = []

    # Y类别个数
    class_categories_num = len(df[y].cat.categories)

    for i in np.arange(0,(len(df.columns) - 1)):

        # 特征类别个数
        features_categories_num = len(df[df.columns[i]].cat.categories)

        Vec = np.zeros(shape=(features_categories_num,class_categories_num))
        for j in np.arange(0,features_categories_num):
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
    H_DA = (P_Di * H_Di).sum()

    return (H_DA)

# 特征A的IV值
def IV(Di_vec,Aik_vec):
    IV = np.zeros(len(Aik_vec))
    for i in np.arange(0,len(Aik_vec)):
        p_i = (Aik_vec[i].sum()) / (Di_vec.sum())
        if p_i != 0:
            IV[i] = p_i * log(p_i,2)
        else:
            IV[i] = 0
    IV = -(IV.sum())

    return IV

# 特征A的信息增益比
def gain_ratio(Di_vec,Aik_vec):
    gain_A = entropy(Di_vec) - con_entroy(Di_vec,Aik_vec)
    IV_A = IV(Di_vec,Aik_vec)

    if IV_A != 0:
        gain_ratio_A = gain_A/IV_A
    else:
        gain_ratio_A = 0

    return (gain_ratio_A)


# 计算每个特征的信息增益比，并取最大值
def gain_ratio_max(df,y):
    gain_ratio_vec = np.zeros(shape=((len(df.columns) - 1),1))

    feature_split_num = feature_split(df,y)

    # Y类别个数
    Di_vec = np.array(df[y].value_counts())

    # 计算各特征信息增益
    for i in np.arange(0,len(feature_split_num)):
        gain_ratio_vec[i] = gain_ratio(Di_vec,feature_split_num[i])

    # 选取信息增益最大的特征
    return [df.columns[gain_ratio_vec.argmax()],gain_ratio_vec.max()]


# 训练 ---------------------------------------------------------
def Decision_Tree(DTree,y,delta):
    for key,value in DTree.items():

        # DTree = currentTree

        subTree = {}

        # value = DTree['SibSp = 1']

        if isinstance(value,pd.DataFrame):
            df = value

            # 判断是否信息增益达到阈值
            if (len(df.columns) - 1) >= 1 and gain_ratio_max(df,y)[1] >= delta:
                split_feature_name = gain_ratio_max(df,y)[0]

                for cat in df[split_feature_name].cat.categories:

                    # cat = 1

                    df_split_temp = df[df[split_feature_name] == cat].drop(split_feature_name,axis=1)
                    description = ' '.join([str(split_feature_name),'=',str(cat)])

                    if (len(df_split_temp[y].unique()) != 1) and (df_split_temp.empty != True):

                        currentTree = {description: df_split_temp}
                        currentValue = Decision_Tree(currentTree,y,delta)

                        subTree.update(currentValue)

                    else:

                        if (len(df_split_temp[y].unique()) == 1):
                            leaf_node = df_split_temp[y].values[0]

                        if (df_split_temp.empty == True):
                            leaf_node = df[y].value_counts().argmax() # 分裂前的最多类别

                        subTree.update({description: leaf_node})

            elif (len(df.columns) - 1) < 1:
                leaf_node = df[y].value_counts().argmax() # 如果只剩Y一列，取当前多的最多类别

                subTree = leaf_node

            elif gain_ratio_max(df,y)[1] < delta:
                leaf_node = df[y].value_counts().argmax() # 分裂前的最多类别

                subTree = leaf_node

            DTree[key] = subTree

        else:
            print("Done!")

    return DTree


def C45(data,y,delta=0.005):

    # 标准化数据集
    data = order_Y(data,y)
    y = 'label'

    DTree = {}

    if gain_ratio_max(data,y)[1] >= delta:
        split_feature_name = gain_ratio_max(data,y)[0]

        # 初次分裂
        for cat in data[split_feature_name].cat.categories:
            # print(cat)

            # cat = 1
            data_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
            description = ' '.join([str(split_feature_name),'=',str(cat)])

            currentValue = data_split_temp

            if gain_ratio_max(data_split_temp,y)[1] < delta:
                currentValue = data_split_temp[y].value_counts().argmax()

            if (len(data_split_temp[y].unique()) == 1):
                currentValue = data_split_temp[y].unique()[0]

            if data_split_temp.empty == True:
                currentValue = data[y].value_counts().argmax() # 分裂前的最多类别

            currentTree = {description: currentValue}
            DTree.update(currentTree)

    return Decision_Tree(DTree,y,delta)


# 预测 ---------------------------------------------------------
def most_leaf_node(tree, leaf_node_list):

    for value in tree.values():
        if isinstance(value,dict):
            most_leaf_node(value, leaf_node_list)
        else:
            leaf_node_list.append(value)

    return max(set(leaf_node_list),key=leaf_node_list.count)


def most_class(tree):
    leaf_node_list = []
    return most_leaf_node(tree,leaf_node_list)


def C45_predict_one(DTree,row_data):
    for keys,values in DTree.items():
        T_key = keys
        T_value = values

        T_key_list = re.split('(=|<|<=|>|>=|!=)',T_key)
        split_feature = T_key_list[0].strip()
        split_feature_oper = T_key_list[1].strip()
        split_feature_value = T_key_list[2].strip()

        if str(row_data[split_feature]) == split_feature_value:
            if isinstance(T_value,dict):
                return C45_predict_one(T_value,row_data)
            else:
                return T_value


def C45_predict(DTree,new_data):
    predict_Y = []

    most_leaf = most_class(DTree)

    for row_data in new_data.iterrows():

        row_data_series = row_data[1]

        pre_y = C45_predict_one(DTree,row_data_series)
        # if pre_y == None:
        #     pre_y = most_leaf     # 问题已修复，应该不会出现NONE了！【待修改】
        #     print('fix!')
        predict_Y.append(pre_y)

    return (predict_Y)

# --------------------------------- 测试 -------------------------------------- #
# 1.西瓜数据集

data = pd.read_csv('data/watermelon2.0.csv')
for i in np.arange(len(data.columns)):
    data.ix[:,i] = data.ix[:,i].astype('category')
data = data.drop(['id'],axis=1)

model_DT = C45(data=data,y='haogua',delta=0.005)
pre_Y = C45_predict(model_DT,data)

# 2.Kaggle Titanic Data

# 读取数据
train = pd.read_csv('Data/train_fixed.csv')
test = pd.read_csv('Data/test_fixed.csv')

# 转为分类型变量
for i in np.arange(len(train.columns)):
    train.ix[:,i] = train.ix[:,i].astype('category')
for i in np.arange(len(test.columns)):
    test.ix[:,i] = test.ix[:,i].astype('category')

# 分割数据
train_train,train_test = train_test_split(train,test_size=0.4,random_state=0)

# 训练
start = time.clock()
model_DT = C45(data=train_train,y='Survived',delta=0.005)
elapsed = (time.clock() - start)
# 预测
start = time.clock()
pre_Y = C45_predict(model_DT,train_test)
elapsed = (time.clock() - start)

# AUC
pre_dt = pd.DataFrame({'Y': train_test['Survived'],'pre_Y': pre_Y})
pre_dt['Y'].cat.categories
pre_dt.ix[:,'pre_Y'] = pre_dt.ix[:,'pre_Y'].astype('category')
pre_dt['pre_Y'].cat.categories
roc_auc_score(pre_dt.Y,pre_dt.pre_Y)

# Submit
pre_Y = C45_predict(model_DT,test)
submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
submit.ix[:,'Survived'] = submit.ix[:,'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('E:/GitHub/Algorithms/Result/submit.csv',index=False)
