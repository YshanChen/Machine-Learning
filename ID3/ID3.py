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
df.age = np.where(df.age == "青年",np.where(df.age == "中年",2,3),1)
df.loan = np.where(df.loan == "一般",1,
                   np.where(df.loan == "好",2,3))

df.work = np.where(df.work == "是",1,0)
df.hourse = np.where(df.hourse == "是",1,0)
df['class'] = np.where(df['class'] == "是",1,0)

for i in np.arange(len(df.columns)):
    df.ix[:,i] = df.ix[:,i].astype('category')

df = df[['class','work','hourse','loan','age']]


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


# 训练 ---------------------------------------------------------

key = 'Sex = 1'
value = DTree[key]

DTree = currentTree
key = 'Fare = 3'
value = DTree[key]

DTree = currentTree
key = 'Age = 4'
value = DTree[key]

DTree = currentTree
key = 'Embarked = 1'
value = DTree[key]


DTree = currentTree
key = 'Pclass = 2'
value = DTree[key]

DTree = currentTree
key = 'SibSp = 1'
value = DTree[key]

DTree = currentTree
key = 'Parch = 0'
value = DTree[key]


def Decision_Tree(DTree,y,delta,max_class_in_D,par_description = ''):
    for key,value in DTree.items():

        print([key,value])
        print('--------------------------')

        subTree = {}

        if isinstance(value,pd.DataFrame):
            df = value

            # 判断是否信息增益达到阈值
            if (len(df.columns) - 1) >= 1 and gain_max(df,y)[1] >= delta:
                print('到这里了？分裂')
                print("满足delta和特征数量")

                split_feature_name = gain_max(df,y)[0]

                for cat in np.unique(df[split_feature_name]):

                    # cat = 1

                    df_split_temp = df[df[split_feature_name] == cat].drop(split_feature_name,axis=1)
                    description = ' '.join([str(split_feature_name),'=',str(cat)])
                    print(description)
                    par_description = description

                    print("有没有？！")
                    if (len(df_split_temp[y].unique()) != 1) and (df_split_temp.empty != True):

                        print("继续分裂！")

                        currentTree = {description: df_split_temp}
                        currentValue = Decision_Tree(currentTree,y,delta,max_class_in_D,par_description)

                        print("没有出现就上面错了！")
                        print(currentValue)
                        subTree.update(currentValue)

                    else:
                        print("没有分裂！")

                        if (len(df_split_temp[y].unique()) == 1):
                            print('Leaf Node：唯一类！')
                            leaf_node = df_split_temp[y].values[0]

                        if (df_split_temp.empty == True):
                            print('Leaf Node：空集！')
                            leaf_node = max_class_in_D

                        print("两个都不满足？！")
                        print({description: leaf_node})
                        subTree.update({description: leaf_node})

            elif (len(df.columns) - 1) < 1:
                print('没得特征分裂了！')
                leaf_node = df[y].values[0]

                print(leaf_node)
                subTree = leaf_node

            elif gain_max(df,y)[1] < delta:
                print("不满足delta")
                leaf_node = max_class_in_D

                print(leaf_node)
                subTree = leaf_node

            DTree[key] = subTree

        else:
            print("Data is not a DataFrame!")

    print('====================================================')
    return DTree


def ID3(data,y,delta=0.005):
    # 标准化数据集
    data = order_Y(data,y)
    y = 'label'

    DTree = {}

    max_class_in_D = data[y].value_counts().argmax()  # D中实例最大的类

    if gain_max(data,y)[1] >= delta :
        split_feature_name = gain_max(data,y)[0]

        # 初次分裂
        for cat in np.unique(data[split_feature_name]):

            # cat = 1
            data_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
            description = ' '.join([str(split_feature_name),'=',str(cat)])

            currentValue = data_split_temp

            if gain_max(data_split_temp,y)[1] < delta:
                currentValue = max_class_in_D

            if (len(data_split_temp[y].unique()) == 1):
                currentValue = data[y].values[0]

            if data_split_temp.empty == True:
                currentValue = max_class_in_D

            currentTree = {description: currentValue}
            DTree.update(currentTree)

    return Decision_Tree(DTree,y,delta,max_class_in_D)


# 预测 ---------------------------------------------------------
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

        row_data_series = row_data[1]
        predict_Y.append(ID3_predict_one(DTree,row_data_series))

    return (predict_Y)


# --------------------------------- 测试 -------------------------------------- #
# Kaggle Titanic Data
data = pd.read_csv('Data/train.csv')

data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

# 1
data.ix[:,'Survived'] = data.ix[:,'Survived'].astype('category')

# 2
data['Pclass'].value_counts()
data.ix[:,'Pclass'] = data.ix[:,'Pclass'].astype('category')

# 3
data['Sex'].value_counts()
data.Sex = np.where(data.Sex == "male",1,2)
data.ix[:,'Sex'] = data.ix[:,'Sex'].astype('category')

# 4
data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data.Age = np.where(data.Age <= 10,1,
                    np.where(data.Age <= 20,2,
                             np.where(data.Age <= 30,3,
                                      np.where(data.Age <= 40,4,
                                               np.where(data.Age <= 50,5,6)))))
data.ix[:,'Age'] = data.ix[:,'Age'].astype('category')

# 5
data['SibSp'].value_counts()
data.ix[:,'SibSp'] = data.ix[:,'SibSp'].astype('category')

# 6
any(pd.isnull(data['Parch']))
data['Parch'].value_counts()
data.ix[:,'Parch'] = data.ix[:,'Parch'].astype('category')

# 7
pd.isnull(['Fare'])
data['Fare'] = data['Fare'].fillna(np.mean(data['Fare']))

data.Fare.describe()

data.Fare = np.where(data.Fare <= 7,1,
                     np.where(data.Fare <= 15,2,
                              np.where(data.Fare <= 32,3,
                                       np.where(data.Fare <= 50,4,
                                                np.where(data.Fare <= 80,5,6)))))
data.ix[:,'Fare'] = data.ix[:,'Fare'].astype('category')

# 8
any(pd.isnull(data['Embarked']))
data['Embarked'] = data['Embarked'].fillna(data.Embarked.value_counts()[0])
data.Embarked = np.where(data.Embarked == 'S',1,
                         np.where(data.Embarked == 'C',2,3))
data.ix[:,'Embarked'] = data.ix[:,'Embarked'].astype('category')

# ----------------
model_DT = ID3(data=df,y='class',delta=0.005)
pre_Y = ID3_predict(model_DT,df)

td = data

model_DT = ID3(data=data,y='Survived',delta=0.005)
pre_Y = ID3_predict(model_DT,td)


