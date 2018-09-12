# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time

'''
重构ID3算法，面向对象形式

'''

class DTree(object):

    def __init__(self,  algorithm, delta=0.005):
        self.params = {}
        self.params['delta'] = delta

        if algorithm in ['ID3', 'C4.5', 'CART']:
            self.algorithm = algorithm
        else:
            raise ValueError('algorithm must be [''ID3', 'C4.5', 'CART'']')

    def fit(self, data, y):

        if self.algorithm == 'ID3':
            self.DTree = ID3(data=data, y=y, delta=self.params['delta'])

    def predict(self, new_data):

        if self.algorithm == 'ID3':
            return ID3_predict(DTree=self.DTree, new_data=new_data)

clf = DTree(algorithm='ID3', delta=0.001)
clf.fit(data=train_train, y='Survived')
clf.DTree
clf.predict(new_data=train_test)

# 信息增益算法 -----------------------------------------------------
def order_Y(data, y):
    df = data.copy()
    df['label'] = df[y]
    df = df.drop([y],axis=1)
    return df

# 特征分裂向量 （计算每个特征的每个取值对应的Y类别的个数）
def feature_split(data, X, y):
    feature_split_dic = {}

    # X个数
    X_num = len(X)

    # Y类别 & 个数
    y_classes = data[y].cat.categories
    y_class_num = len(y_classes)

    # 计算每个特征的每个取值对应的Y类别的个数
    for feature_name in X:

        # 特征A的取值 & 个数
        feature_values = data[feature_name].cat.categories
        feature_values_num = len(feature_values)

        # 特征的每个取值对应的Y类别的个数
        Vec = np.zeros(shape=(feature_values_num, y_class_num))
        for feature_value_index, feature_value in enumerate(feature_values):
            for y_class_index, y_class in enumerate(y_classes):
                count_number = ((data[feature_name] == feature_value) & (data[y] == y_class)).sum()
                Vec[feature_value_index, y_class_index] = count_number

        # 打印:分裂特征 & 取值对应类别个数
        # print('Feature Split Name : ', feature_name)
        # print('Feature Class Number : ', Vec)
        # print('--------------------------')

        feature_split_dic[feature_name] = Vec

    return feature_split_dic


# 数据集D的经验熵（empirical entropy）
def entropy(Di_vec):
    # Di_vec => np.array
    if isinstance(Di_vec, dict):
        Di_vec = np.array(list(Di_vec.values()))

    # 总集合的个数
    D_num = Di_vec.sum()

    # 计算：子集个数/总集个数
    if D_num == 0:
        p_vec = np.zeros(shape=(len(Di_vec)))
    else:
        p_vec = Di_vec / D_num

    # 计算：empirical entropy
    h_vec = np.array([])
    for p in p_vec:
        if p != 0:
            h = p * log(p, 2)
            h_vec = np.append(h_vec, h)
        else:
            h_vec = np.append(h_vec, 0)  # Todo: 对于不存在特征取值的情况，如何处理
    H = -(h_vec.sum())

    return (H)

# 特征A对数据集D的条件熵
def conditional_entroy(Di_vec, Aik_vec):
    H_Di = np.array([])
    P_Di = np.array([])
    for D_i in Aik_vec:
        print(D_i)
        H_Di = np.append(H_Di,entropy(D_i))
        P_Di = np.append(P_Di,(D_i.sum() / Di_vec.sum()))
    H_DA = (H_Di * P_Di).sum()

    return (H_DA)


# 特征A的信息增益
def gain(Di_vec, Aik_vec):
    gain_A = entropy(Di_vec) - conditional_entroy(Di_vec,Aik_vec)

    return (gain_A)


# 计算每个特征的信息增益，并取最大值
def gain_max(data, X, y):
    # 特征变量个数
    X_number = len(X)

    # 每个特征的信息增益
    gain_dic = dict.fromkeys(X, 0)

    # 计算每个特征的每个取值对应的Y类别的个数
    feature_split_dic = feature_split(data, X, y)

    # Y类别个数
    Di_dic = dict(data[y].value_counts())

    # 计算各特征的信息增益
    for feature_name in feature_split_dic.keys():
        print(feature_name)
        print('---------------')
        gain_vec[i] = gain(Di_vec=Di_dic, Aik_vec=feature_split_vec[feature_name])

    # 选取信息增益最大的特征
    return [data.columns[gain_vec.argmax()],gain_vec.max()]


# 训练 ---------------------------------------------------------
def Decision_Tree(DTree,y,delta):
    for key,value in DTree.items():

        subTree = {}

        if isinstance(value,pd.DataFrame):
            df = value

            # 判断是否信息增益达到阈值
            if (len(df.columns) - 1) >= 1 and gain_max(df,y)[1] >= delta:
                split_feature_name = gain_max(df,y)[0]

                for cat in df[split_feature_name].cat.categories:

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

            elif gain_max(df,y)[1] < delta:
                leaf_node = df[y].value_counts().argmax() # 分裂前的最多类别

                subTree = leaf_node

            DTree[key] = subTree

        else:
            print("Done!")

    return DTree


def ID3(X, y, delta=0.005):
    # Data
    data = pd.concat([X, y], axis=1)

    # 标准化数据集
    data = order_Y(data, y)
    y = 'label'

    # X
    X = data.drop([y], axis=1).columns

    DTree = {}

    if gain_max(data, y)[1] >= delta:
        split_feature_name = gain_max(data, X, y)[0]

        # 初次分裂
        for cat in data[split_feature_name].cat.categories:
            # print(cat)

            # cat = 1
            data_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
            description = ' '.join([str(split_feature_name),'=',str(cat)])

            currentValue = data_split_temp

            if gain_max(data_split_temp,y)[1] < delta:
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

    most_leaf = most_class(DTree)

    for row_data in new_data.iterrows():

        row_data_series = row_data[1]

        pre_y = ID3_predict_one(DTree,row_data_series)
        # if pre_y == None:
        #     pre_y = most_leaf     # 问题已修复，应该不会出现NONE了！【待修改】

        predict_Y.append(pre_y)

    return (predict_Y)

# --------------------------------- 测试 -------------------------------------- #
# 1.西瓜数据集
data = pd.read_csv('data/watermelon2.0.csv')
for i in np.arange(len(data.columns)):
    data.ix[:,i] = data.ix[:,i].astype('category')
data = data.drop(['id'],axis=1)

model_DT = ID3(data=data, y='haogua', delta=0.005)

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
model_DT = ID3(data=train_train,y='Survived',delta=0.005)
elapsed = (time.clock() - start)
# 预测
start = time.clock()
pre_Y = ID3_predict(model_DT,train_test)
elapsed = (time.clock() - start)

# AUC
pre_dt = pd.DataFrame({'Y': train_test['Survived'],'pre_Y': pre_Y})
pre_dt['Y'].cat.categories
pre_dt.ix[:,'pre_Y'] = pre_dt.ix[:,'pre_Y'].astype('category')
pre_dt['pre_Y'].cat.categories
roc_auc_score(pre_dt.Y,pre_dt.pre_Y)

# Submit
pre_Y = ID3_predict(model_DT,test)
submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
submit.ix[:,'Survived'] = submit.ix[:,'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('E:/GitHub/Algorithms/Result/submit.csv',index=False)
