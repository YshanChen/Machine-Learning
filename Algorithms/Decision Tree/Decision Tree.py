# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time

'''
统一决策树算法框架

delta: 即sklearn库中的min_impurity_split, 节点划分最小不纯度，如果不纯度小于该值则停止分裂。

'''

class DTree(object):

    def __init__(self,  method, delta=0.005):
        self.params = {}
        self.params['delta'] = delta

        if method in ['ID3', 'C4.5', 'CART']:
            self.method = method
        else:
            raise ValueError('method must be [''ID3', 'C4.5', 'CART'']')

    def fit(self, X, y):
        if self.method == 'ID3':
            self.DTree = ID3(X=X, y=y, method=self.method, delta=self.params['delta'])
        if self.method == 'C4.5':
            self.DTree = C4_5(X=X, y=y, method=self.method, delta=self.params['delta'])

    def predict(self, new_data):
        return predict(DTree=self.DTree, new_data=new_data)

# 信息增益算法 -----------------------------------------------------
# 特征分裂向量 （计算每个特征的每个取值对应的Y类别的个数）
def feature_split(data, y):
    feature_split_dic = {}

    # X个数
    X = data.drop([y], axis=1).columns
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
        feature_values_dict = {}
        for feature_value in feature_values:
            Vec = np.zeros(shape=(1, y_class_num))[0]
            for y_class_index, y_class in enumerate(y_classes):
                count_number = ((data[feature_name] == feature_value) & (data[y] == y_class)).sum()
                Vec[y_class_index] = count_number
            feature_values_dict[feature_value] = Vec

        # 打印:分裂特征 & 取值对应类别个数
        # print('Feature Split Name : ', feature_name)
        # print('Feature Class Number : ', Vec)
        # print('--------------------------')

        feature_split_dic[feature_name] = feature_values_dict

    return feature_split_dic


# 数据集D的经验熵
def entropy(Di_dic):
    # Di_dic => np.array
    if isinstance(Di_dic, dict):
        Di_dic = np.array(list(Di_dic.values()))

    # 总集合的个数
    D_num = Di_dic.sum()

    # 计算：子集个数/总集个数
    if D_num == 0:
        p_vec = np.zeros(shape=(len(Di_dic)))
    else:
        p_vec = Di_dic / D_num

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
def conditional_entroy(Di_dic, Aik_vec):
    # Di_dic => np.array
    if isinstance(Di_dic, dict):
        Di_dic = np.array(list(Di_dic.values()))

    H_Di = np.array([])
    P_Di = np.array([])
    for Aik in Aik_vec.keys():
        H_Di = np.append(H_Di,entropy(Aik_vec[Aik]))
        P_Di = np.append(P_Di,(Aik_vec[Aik].sum() / Di_dic.sum()))

    # 判断根据特征取值划分的集合的样本个数/总集合样本个数的和为1
    if abs(1-P_Di.sum()) >= 0.0001:
        raise ValueError("P_Di sum is not 1 !")

    H_DA = (H_Di * P_Di).sum()

    return (H_DA)

# 数据集关于特征A的值的熵
def HaD(Di_dic, Aik_vec):

    HaD_dic = dict.fromkeys(list(Aik_vec.keys()), 0)
    for a_i in Aik_vec.keys():
        p_i = Aik_vec[a_i].sum() / sum(Di_dic.values())

        if p_i != 0:
            HaD_dic[a_i] = p_i * log(p_i,2)
        else:
            HaD_dic[a_i] = 0

    HaD = -sum(HaD_dic.values())

    return HaD

# 特征A的信息增益/信息增益比
def gain(Di_dic, Aik_vec, method):
    if method == 'ID3':
        gain_A = entropy(Di_dic) - conditional_entroy(Di_dic,Aik_vec)
        gain = gain_A

    elif method == 'C4.5':
        gain_A = entropy(Di_dic) - conditional_entroy(Di_dic, Aik_vec)
        HaD_ = HaD(Di_dic, Aik_vec)
        if HaD_ != 0:
            gain_ratio_A = gain_A / HaD_
        else:
            gain_ratio_A = 0
        gain = gain_ratio_A

    return gain

# 计算每个特征的信息增益/信息增益比，并取最大值
def gain_max(data, y, method):
    # 特征变量个数
    X = data.drop([y], axis=1).columns
    X_number = len(X)

    # 每个特征的信息增益
    gain_dic = dict.fromkeys(X, 0)

    # 计算每个特征的每个取值对应的Y类别的个数
    feature_split_dic = feature_split(data, y)

    # Y类别个数
    Di_dic = dict(data[y].value_counts())

    # 计算各特征的信息增益
    if gain_dic.keys() != feature_split_dic.keys():
        raise ValueError("features are wrong !")

    for feature_name in gain_dic.keys():
        gain_dic[feature_name] = gain(Di_dic=Di_dic, Aik_vec=feature_split_dic[feature_name], method=method)

    # 选取信息增益最大的特征
    max_gain_feature = max(gain_dic, key=gain_dic.get)
    max_gain = gain_dic[max_gain_feature]

    return [max_gain_feature, max_gain]


# 训练 ---------------------------------------------------------
def Decision_Tree(DTree, y, method, delta):
    for key, value in DTree.items():
        print(key)
        # key = key
        # value = value value = DTree[key]

        # 子树
        subTree = {}

        # 判断是否为叶子结点
        if isinstance(value, pd.DataFrame):
            data = value

            # 特征变量X
            X = data.drop([y], axis=1).columns

            # 判断：信息增益是否达到阈值 & 是否特征变量>=1
            gain_list = gain_max(data, y, method)
            if len(X) > 1 and gain_list[1] >= delta:
                split_feature_name = gain_list[0]

                for cat in data[split_feature_name].cat.categories:

                    # 分裂  cat = 0
                    df_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
                    description = ' '.join([str(split_feature_name),'=',str(cat)])

                    # 停止条件判断： 分裂后类别是否唯一 & 分裂后是否为空置
                    if (len(df_split_temp[y].unique()) != 1) and (df_split_temp.empty != True):
                        currentTree = {description: df_split_temp}
                        currentValue = Decision_Tree(currentTree, y, method, delta) # 递归
                        subTree.update(currentValue)

                    else:
                        # 分裂后类别唯一，叶子结点为该类别 (需要分裂)
                        if (len(df_split_temp[y].unique()) == 1):
                            leaf_node = df_split_temp[y].values[0]

                        # 分裂后为空置，叶子结点为分裂前样本最多的类别 (不分裂)
                        if (df_split_temp.empty == True):
                            leaf_node = data[y].value_counts().idxmax() # 分裂前的最多类别 # todo: 不需要分裂，是否需要放到后面统一格式

                        subTree.update({description: leaf_node})

            # 停止条件判断：特征变量<=1，取样本最多的类别 (不分裂)
            elif len(X) <= 1:
                print('特征变量<=1')
                leaf_node = data[y].value_counts().idxmax()
                subTree = leaf_node

            # 停止条件判断：分裂后最大信息增益小于阈值，取分裂前样本最多的类别 (不分裂)
            elif gain_max(data, y, method)[1] < delta:
                print('分裂后最大信息增益小于阈值')
                leaf_node = data[y].value_counts().idxmax() # 分裂前的最多类别
                subTree = leaf_node

            DTree[key] = subTree

        else:
            print("Done!")

    return DTree

def ID3(X, y, method, delta=0.005):
    # Data
    data = pd.concat([X, y], axis=1).rename(str, columns={y.name:'label'})

    # define y
    y = 'label'

    # X
    X = data.drop([y], axis=1).columns

    DTree = {}

    if gain_max(data, y, method)[1] >= delta:
        split_feature_name = gain_max(data, y)[0]

        # 初次分裂
        for cat in data[split_feature_name].cat.categories:
            # print(cat)

            # 分裂         cat = 1
            data_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
            description = ' '.join([str(split_feature_name),'=',str(cat)])

            # 分裂后数据集
            currentValue = data_split_temp

            # 停止分裂判断：如果分裂后最大信息增益依然小于delta，则停止分裂，叶子结点为当前数据集下样本最多的类别
            if gain_max(data_split_temp, y)[1] < delta:
                currentValue = data_split_temp[y].value_counts().idxmax()

            # 停止分裂判断：如果分裂后类别唯一，则停止分裂，叶子结点为该类别
            if (len(data_split_temp[y].unique()) == 1):
                currentValue = data_split_temp[y].unique()[0]

            # 停止分裂判断：如果分裂后为空集，则停止分裂，叶子结点为分裂前的最多类别
            if data_split_temp.empty == True:
                currentValue = data[y].value_counts().idxmax() # 分裂前的最多类别

            # 绘制树结构字典
            currentTree = {description: currentValue}
            DTree.update(currentTree)

    return Decision_Tree(DTree=DTree, y=y, method=method, delta=delta)

def C4_5(X, y, method, delta=0.005):
    # Data
    data = pd.concat([X, y], axis=1).rename(str, columns={y.name:'label'})

    # define y
    y = 'label'

    # X
    X = data.drop([y], axis=1).columns

    DTree = {}

    gain_list = gain_max(data, y, method)
    if gain_list[1] >= delta:
        split_feature_name = gain_list[0]

        # 初次分裂
        for cat in data[split_feature_name].cat.categories:
            # print(cat)

            # 分裂         cat = 1
            data_split_temp = data[data[split_feature_name] == cat].drop(split_feature_name,axis=1)
            description = ' '.join([str(split_feature_name),'=',str(cat)])

            # 分裂后数据集
            currentValue = data_split_temp

            # 停止分裂判断 (1)：如果分裂后最大信息增益依然小于delta，则停止分裂，叶子结点为当前数据集下样本最多的类别
            if gain_max(data_split_temp, y, method)[1] < delta:
                currentValue = data_split_temp[y].value_counts().idxmax()

            # 停止分裂判断 (2)：如果分裂后类别唯一，则停止分裂，叶子结点为该类别
            if len(data_split_temp[y].unique()) == 1:
                currentValue = data_split_temp[y].unique()[0]

            # 停止分裂判断 (3)：如果分裂后为空集，则停止分裂，叶子结点为分裂前的最多类别
            if data_split_temp.empty == True:
                currentValue = data[y].value_counts().idxmax() # 分裂前的最多类别

            # 绘制树结构字典
            currentTree = {description: currentValue}
            DTree.update(currentTree)

    return Decision_Tree(DTree=DTree, y=y, method=method, delta=delta)


# 预测 ---------------------------------------------------------
# 获取样本最多的类别
def most_leaf_node(tree, leaf_node_list):
    for value in tree.values():
        if isinstance(value, dict):
            most_leaf_node(value, leaf_node_list)
        else:
            leaf_node_list.append(value)
    return leaf_node_list
    # return max(set(leaf_node_list), key=leaf_node_list.count)

def predict_one_by_one(DTree, row_data):
    for keys,values in DTree.items():
        T_key = keys
        T_value = values

        T_key_list = re.split('(=|<|<=|>|>=|!=)', T_key)
        split_feature = T_key_list[0].strip()
        split_feature_oper = T_key_list[1].strip()
        split_feature_value = T_key_list[2].strip()

        # ID3 非二叉树
        if str(row_data[split_feature]) == split_feature_value: # 符合就继续往下走
            if isinstance(T_value, dict):  # 还有分支情况
                return predict_one_by_one(DTree=T_value, row_data=row_data)
            else:  # 叶子节点情况
                return T_value

def predict(DTree, new_data):
    predict_Y = pd.Series([])

    # 样本最多的类别
    leaf_node_list = []
    most_leaf = most_leaf_node(DTree, leaf_node_list)

    # 逐条样本预测
    for row_index, row_data in new_data.iterrows():
        pre_y = predict_one_by_one(DTree, row_data)
        # if pre_y == None:
        #     pre_y = most_leaf     # 出现NONE，强制赋值为初始样本样本数最多的类别！【待修改】
        predict_Y[row_index] = pre_y

    return predict_Y

# --------------------------------- 测试 -------------------------------------- #
# 1.西瓜数据集
data = pd.read_csv('data/watermelon2.0.csv')
for i in np.arange(len(data.columns)):
    data.iloc[:,i] = data.iloc[:,i].astype('category')
data = data.drop(['id'],axis=1)

# # 增加连续型变量
# data['density'] = [0.243, 0.245, 0.343, 0.36, 0.403, 0.437, 0.481, 0.556, 0.593, 0.608, 0.634, 0.639, 0.657, 0.666, 0.697, 0.719, 0.774]

X = data.drop(['haogua'], axis=1)
y = data['haogua']

clf = DTree(method='C4.5', delta=0.01)
clf.fit(X=X, y=y)
clf.DTree
y_test = clf.predict(new_data=X)

# 2.Kaggle Titanic Data
# 读取数据
train = pd.read_csv('Data/train_fixed.csv')
test = pd.read_csv('Data/test_fixed.csv')

# 转为分类型变量
for i in np.arange(len(train.columns)):
    train.iloc[:,i] = train.iloc[:,i].astype('category')
for i in np.arange(len(test.columns)):
    test.iloc[:,i] = test.iloc[:,i].astype('category')

# 分割数据
train_train, train_test = train_test_split(train,test_size=0.4,random_state=0)

X_train = train_train.drop(['Survived'], axis=1)
y_train = train_train['Survived']
X_test = train_test.drop(['Survived'], axis=1)
y_test = train_test['Survived']

# 分类器
clf = DTree(method='C4.5', delta=0.001)
# 训练
start = time.clock()
clf.fit(X=X_train, y=y_train)
elapsed = (time.clock() - start)
print("Train Model Time : ", elapsed)
# 预测
start = time.clock()
y_test_pred = clf.predict(new_data=X_test)
elapsed = (time.clock() - start)
print("Predict Model Time : ", elapsed)

# AUC
pre_dt = pd.DataFrame({'Y': train_test['Survived'],'pre_Y': y_test_pred})
print('AUC for Test : ', roc_auc_score(pre_dt.Y,pre_dt.pre_Y))

# Submit
pre_Y = clf.predict(new_data=test)
submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
submit.loc[:,'Survived'] = submit.loc[:,'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20180912.csv',index=False)
