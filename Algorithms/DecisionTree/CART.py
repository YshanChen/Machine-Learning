# -*- coding: utf8 -*-
"""
Created on 2018/09/18
@author: Yshan.Chen

Update: 2018/10/23

Commit：
1. 完成基尼指数函数；
2. 完成连续值的处理；
3. 完成生成树；
4. 加入停止条件：
    1) max_depth
    2) min_impurity_split
    3) max_features
    4) min_samples_split
    5) min_samples_leaf

Todo List:
1. 完成回归树
1. 缺失值的处理；1)如何在属性值缺失情况下特征选择？ 2)给定分裂特征，若样本在该特征上缺失，如何对样本进行划分？
2. 树剪枝； CART算法的剪枝与ID3和C4.5不同
3. 预测-并行化
"""

import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import time

class CART(object):
    """
    CART算法
    1. 二叉树结构，所以对于categorical和numeric特征都是二分法
    2. categorical特征需要先onehot encoding

    min_impurity_split: 节点划分最小不纯度，如果不纯度小等于该值则停止分裂 △ Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
    max_features: 划分时考虑的最大特征数 △
    max_depth: 决策树最大深度 △
    min_samples_split: 内部节点再划分所需最小样本数 △
    min_samples_leaf: 叶子节点最少样本数 △
    min_weight_fraction_leaf: 叶子节点最小的样本权重和 (缺失值处理涉及)
    max_leaf_nodes: 最大叶子节点数
    """

    def __init__(self,
                 objective,
                 min_impurity_split=0.005,
                 max_features=0,
                 max_depth=3,
                 min_samples_split=5,
                 min_samples_leaf=1
                 ):
        if objective not in ["regression", "binary"]:
            raise Exception("Error: objective must be \"regression\" or \"binary\" !")
        else:
            self.params = {'objective': objective,
                           'min_impurity_split': min_impurity_split,
                           'max_features':max_features,
                           'max_depth':max_depth,
                           'min_samples_split':min_samples_split,
                           'min_samples_leaf':min_samples_leaf}
            self.DTree = {}

    def fit(self, X, y):
        if self.params['objective'] == 'binary':
            self.DTree = self._Decision_Tree_binary(X=X, y=y)
        if self.params['objective'] == 'regression':
            self.DTree = self._Decision_Tree_regression(X=X, y=y)

    def predict(self, new_data): # 逐条预测，未实现并行化
        if self.DTree == {}:
            raise ValueError('There is no classifier for predicting !')

        predict_Y = pd.Series([])

        # 样本最多的类别
        leaf_node_list = []
        most_leaf = self._most_leaf_node(self.DTree, leaf_node_list)

        # 逐条样本预测
        for row_index, row_data in new_data.iterrows():
            # row_data = new_data.iloc[0, ]
            pre_y = self._predict_one_by_one(DTree=self.DTree, row_data=row_data)
            # if pre_y == None:
            #     pre_y = most_leaf     # 出现NONE，强制赋值为"初始样本"样本数最多的类别！【待修改】
            predict_Y[row_index] = pre_y

        return predict_Y

    # 特征分裂向量
    '''
    # 计算每个特征的每个取值对应的Y类别的个数
    # 对于categorical和numeric特征都是二分法，categorical特征需要先onehot-encoding
    # 先排序，再去重，依次选择两个取值中分数进行二分,大于 or 小等于
    
    Todo: V2.0 缺失值的处理，需要增加样本权重
    
    '''
    def _feature_split(self, data, y):
        feature_split_dic = {}

        # X个数
        X = data.drop([y], axis=1).columns
        X_num = len(X)

        # Y类别 & 个数
        y_classes = data[y].cat.categories
        y_class_num = len(y_classes)

        # 计算每个特征的每个取值对应的Y类别的个数
        for feature_name in X:  # feature_name = 'density'
            # 排序、去重、取所有特征值
            feature_values_series = data[feature_name].sort_values().drop_duplicates(keep='first')

            # 计算：特征的每个划分点对应的Y类别的个数
            feature_values_dict = {}
            for feature_value_1, feature_value_2 in zip(feature_values_series[0:], feature_values_series[1:]):
                feature_values_vec = {'> a':0 , '<= a':0}  # [>a, <=a]
                feature_split_value = round((feature_value_1+feature_value_2)/2, 4)
                Vec_bigger = {}
                Vec_lesser = {}
                for y_class in y_classes:
                    count_number_bigger = ((data[feature_name] > feature_split_value) & (data[y] == y_class)).sum()
                    count_number_lesser = ((data[feature_name] <= feature_split_value) & (data[y] == y_class)).sum()
                    Vec_bigger[y_class] = count_number_bigger
                    Vec_lesser[y_class] = count_number_lesser
                    feature_values_vec['> a'] = Vec_bigger
                    feature_values_vec['<= a'] = Vec_lesser

                feature_values_dict[feature_split_value] = feature_values_vec

            # 打印:分裂特征 & 取值对应类别个数
            # print('Feature Split Name : ', feature_name)
            # print('Feature Class Number : ', feature_values_dict)

            feature_split_dic[feature_name] = feature_values_dict

        return feature_split_dic

    # 数据集D的基尼指数
    def _gini_D(self, Di_dic):
        # 集合的个数
        D_num = sum(Di_dic.values())

        # 计算：数据集的gini
        g_vec = 0
        for C_k in Di_dic:
            # print((C_k, Di_dic[C_k]))
            g_vec = g_vec + (Di_dic[C_k]/D_num)**2
        gini_D = 1 - g_vec

        return gini_D

    # 数据集在特征A下的条件下的基尼指数（选择基尼指数最小的作为最优特征及U最优特征点）
    def _gini_A(self, Di_dic, Aik_dic):
        # 总集合的个数
        D_num = sum(Di_dic.values())

        # 候选划分点
        A_toSelect = Aik_dic.keys()

        gini_A_dic = {}
        for a_i in A_toSelect:
            D_bigger = Aik_dic[a_i]['> a']
            D_lesser = Aik_dic[a_i]['<= a']

            # 数据集在特征A取该划分点的条件下的基尼指数
            gini_set_bigger = self._gini_D(Di_dic=D_bigger)
            gini_set_lesser = self._gini_D(Di_dic=D_lesser)
            # gini_set_bigger = _gini_D(self=[], Di_dic=D_bigger)
            # gini_set_lesser = _gini_D(self=[], Di_dic=D_lesser)
            gini_D_A = (sum(D_bigger.values()) / D_num * gini_set_bigger) + (sum(D_lesser.values()) / D_num * gini_set_lesser)
            gini_A_dic[a_i] = gini_D_A

        # 选取基尼指数最小的划分点为该特征的最优划分点，相应基尼指数为该特征的最优基尼指数
        gini_A_opt = (min(gini_A_dic, key=gini_A_dic.get), min(gini_A_dic.values()))

        return gini_A_opt

    # 计算每个特征的在每个划分点下的基尼指数，选取最小的基尼指数对应的特征以及最优划分点
    def _gini_min(self, data, y):
        X = data.drop([y], axis=1).columns
        X_number = len(X)

        # 每个特征的增益-字典
        gain_dic = dict.fromkeys(X, 0)

        # 计算每个特征的每个取值对应的Y类别的个数
        feature_split_dic = self._feature_split(data, y)
        # feature_split_dic = _feature_split(self=[], data=data, y=y)

        # Y类别个数
        Di_dic = dict(data[y].value_counts())

        # 计算各特征的增益(gini)
        if gain_dic.keys() != feature_split_dic.keys():
            raise ValueError("Error: Features are wrong !")

        for feature_name in gain_dic.keys():  # feature_name = 'chugan_1'
            gain_dic[feature_name] = self._gini_A(Di_dic=Di_dic, Aik_dic=feature_split_dic[feature_name])
            # gain_dic[feature_name] = _gini_A(self=[], Di_dic=Di_dic, Aik_dic=feature_split_dic[feature_name])

        # 选取基尼指数最小的特征
        min_gini = 2 # 基尼小等于2
        min_gini_feature = ''
        min_gini_feature_point = ''
        for feature, value in gain_dic.items():
            if value[1] < min_gini: # Todo:Or <= ?
                min_gini = value[1]
                min_gini_feature = feature
                min_gini_feature_point = value[0]

        # 返回 划分特征, 最优划分点, 最小基尼指数
        return (min_gini_feature, min_gini_feature_point, min_gini)

    # 排除取值唯一的变量
    def _drop_unique_column(self, data):
        del_unique_columns = []
        for col in [x for x in data.columns if x != 'label']:
            if len(data[col].unique()) <= 1:
                del_unique_columns.append(col)
        data = data.drop(del_unique_columns, axis=1)
        return data

    # 训练 ---------------------------------------------------------
    """
    树的生成。
    - 分裂后特征是否还能继续用于分裂问题，取决于分裂后是否取值唯一，是否还有区分能力。
      对于离散型且onehot处理过的分裂特征，分裂后其特征取值唯一，故能够删除；
      对于连续型分裂特征，分裂后依然可能取值不唯一，故可能保留用于继续分裂；
      非分裂特征随着分裂可能也出现取值唯一情况，故每次分裂后均根据区分能力删除取值唯一的特征；
    """
    def _Decision_Tree_binary(self, X, y, DTree={}, depth=0): # X=X; y=y; DTree={}; depth=0
        # 初次分裂
        if DTree == {}:

            # Data
            data = pd.concat([X, y], axis=1).rename(str, columns={y.name: 'label'})
            data['label'] = data['label'].astype('category')
            y = 'label'
            data = self._drop_unique_column(data) # 排除取值唯一的变量
            # data = _drop_unique_column(self=[], data=data)
            X = data.drop([y], axis=1).columns

            # 生成树桩
            DTree = {}
            depth = 0

            # 计算: (最优划分特征，最优划分点，对应的最小基尼指数)
            gini_list = self._gini_min(data=data, y=y)
            # gini_list = _gini_min(self=[], data=data, y=y)
            min_gini_feature = gini_list[0]
            min_gini_feature_point = gini_list[1]
            min_gini = gini_list[2]

            '''
            分裂判断：
            1. 如果不纯度<=阈值,不分裂(min_impurity_split); 
            2. 类别取值唯一,不分裂； 
            3. 用于分裂结点的样本数小于min_samples_split,不分裂；
            4. 分裂后的两个结点的样本个数不满足min_samples_leaf,不分裂；
            5. 树深度>=max_depth,不分裂；
            '''
            if min_gini >= self.params['min_impurity_split'] and \
                    len(data[y].unique()) > 1 and \
                    data.shape[0] > self.params['min_samples_split'] and \
                    (data[min_gini_feature] > min_gini_feature_point).sum() >= self.params['min_samples_leaf'] and \
                    (data[min_gini_feature] <= min_gini_feature_point).sum() >= self.params['min_samples_leaf'] and \
                    depth < self.params['max_depth']:

                # 确定分裂 ---
                splitting_feature = min_gini_feature
                splitting_point = min_gini_feature_point
                depth = depth + 1
                print([splitting_feature, splitting_point])

                # 分裂ing
                for opera in ['>', '<=']: # 分别处理左右两个branch
                    if opera == '>': # 大于分裂点
                        data_split_temp = data[data[splitting_feature] > splitting_point]
                    else: # 小等于分裂点
                        data_split_temp = data[data[splitting_feature] <= splitting_point]

                    data_split_temp = self._drop_unique_column(data_split_temp)  # 分裂后删除取值唯一的特征
                    # data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                    description = ' '.join([str(splitting_feature), opera, str(splitting_point)])

                    # 对于分裂后的结点的处理（判断是否满足叶子结点的条件，是否还要进行分裂）
                    if len(data_split_temp[y].unique()) == 1:  # 1. 如果分裂后类别唯一，则停止分裂。结点为叶子结点，该类别即为输出。
                        currentValue = data_split_temp[y].value_counts().idxmax()
                        currentTree = {description: currentValue}
                        DTree.update(currentTree)

                    elif data_split_temp.shape[1] <= self.params['max_features']+1: # 2. 停止分裂判断：可用于分裂的特征小于最大特征阈值。最大类别即为输出。
                        currentValue = data_split_temp[y].value_counts().idxmax()
                        currentTree = {description: currentValue}
                        DTree.update(currentTree)

                    else: # 分裂后结点非叶子结点，继续分裂
                        currentTree = {description: data_split_temp}
                        sub_subTree = self._Decision_Tree_binary(X=X, y=y, DTree=currentTree, depth=depth)
                        # X=X; y=y; DTree=currentTree; depth=depth
                        DTree.update(sub_subTree)

            # 确定不分裂 -------
            else:
                # 1. 内部结点样本数小等于最小划分样本数阈值
                if data.shape[0] <= self.params['min_samples_split']:
                    print("split_sample <= min_samples_split !")

                # 2. 类别值唯一
                elif len(data[y].unique()) <= 1:
                    print("data[y].unique() <= 1 !")

                # 3. 分裂后叶子结点样本数少于min_samples_leaf, 不分裂
                elif ((data[min_gini_feature] > min_gini_feature_point).sum() < self.params['min_samples_leaf']) or ((data[min_gini_feature] <= min_gini_feature_point).sum() < self.params['min_samples_leaf']):
                    print("samples_leaf < min_samples_leaf !")

                # 4. 小等于基尼指数阈值
                elif min_gini < self.params['min_impurity_split']:
                    print("initial min_gini < min_impurity_split !")

                # 5. 最大树深度
                elif depth >= self.params['max_depth']:
                    print("depth > max_depth !")

        # 第二次及之后的分裂
        else:
            for key, value in DTree.items():
                print(key)
                print("-------------------------------")
                key = key # key = 'wenli_1 > 0.5'  key = 'Age_1 <= 0.5'
                value = value  # value = DTree[key]

                # 子树
                subTree = {}

                # 判断是否为叶子结点
                if isinstance(value, pd.DataFrame):
                    data = value    # data = value 已经排除了之前用过的且值为一的变量。

                    # 特征变量X
                    X = data.drop([y], axis=1).columns

                    # 计算划分特征，最优划分点，最小基尼指数
                    gini_list = self._gini_min(data, y)
                    # gini_list = _gini_min(self = [], data=data, y=y)
                    min_gini_feature = gini_list[0]
                    min_gini_feature_point = gini_list[1]
                    min_gini = gini_list[2]

                    '''
                    分裂判断：
                    1. 如果不纯度<=阈值,不分裂(min_impurity_split); 
                    2. 类别取值唯一,不分裂； 
                    3. 用于分裂结点的样本数小于min_samples_split,不分裂；
                    4. 分裂后的两个结点的样本个数不满足min_samples_leaf,不分裂；
                    5. 树深度>=max_depth,不分裂；
                    '''
                    if min_gini >= self.params['min_impurity_split'] and len(data[y].unique()) > 1 and data.shape[0] > \
                            self.params['min_samples_split'] and (
                            data[min_gini_feature] > min_gini_feature_point).sum() >= self.params[
                        'min_samples_leaf'] and (data[min_gini_feature] <= min_gini_feature_point).sum() >= self.params[
                        'min_samples_leaf'] and depth < self.params['max_depth']:

                        splitting_feature = min_gini_feature
                        splitting_point = min_gini_feature_point

                        # 确定分裂 --------------
                        depth = depth + 1
                        print([splitting_feature, splitting_point])

                        for opera in ['>', '<=']: # 分别处理左右两个branch
                            if opera == '>':  # 大于分裂点
                                data_split_temp = data[data[splitting_feature] > splitting_point]
                            else:  # 小等于分裂点
                                data_split_temp = data[data[splitting_feature] <= splitting_point]

                            data_split_temp = self._drop_unique_column(data_split_temp)  # 分裂后删除取值唯一的特征
                            # data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                            description = ' '.join([str(splitting_feature), opera, str(splitting_point)])

                            # 对于分裂后的结点的处理（判断是否满足叶子结点的条件，是否还要进行分裂）
                            if len(data_split_temp[y].unique()) == 1:  # 1. 如果分裂后类别唯一，则停止分裂。结点为叶子结点，该类别即为输出。
                                currentValue = data_split_temp[y].value_counts().idxmax()
                                currentTree = {description: currentValue}
                                subTree.update(currentTree)

                            elif data_split_temp.shape[1] <= self.params[
                                'max_features'] + 1:  # 2. 停止分裂判断：可用于分裂的特征小于最大特征阈值。最大类别即为输出。
                                currentValue = data_split_temp[y].value_counts().idxmax()
                                currentTree = {description: currentValue}
                                subTree.update(currentTree)

                            elif depth >= self.params['max_depth']:  # 3. 树深度达到最大深度，则停止分裂: 叶子结点为样本最多的类别
                                currentValue = data_split_temp[y].value_counts().idxmax()
                                currentTree = {description: currentValue}
                                subTree.update(currentTree)

                            else:  # 分裂后结点非叶子结点，继续分裂
                                currentTree = {description: data_split_temp}
                                sub_subTree = self._Decision_Tree_binary(X=X, y=y, DTree=currentTree, depth=depth)
                                subTree.update(sub_subTree)

                    # 确定不分裂 结点作为叶子结点 -------
                    else:
                        # 1. 内部结点样本数小等于最小划分样本数阈值
                        if data.shape[0] <= self.params['min_samples_split']:
                            subTree = data[y].value_counts().idxmax()

                        # 2. 类别值唯一
                        elif len(data[y].unique()) <= 1:
                            subTree = data[y].value_counts().idxmax()

                        # 3. 分裂后叶子结点样本数少于min_samples_leaf, 不分裂
                        elif ((data[min_gini_feature] > min_gini_feature_point).sum() < self.params['min_samples_leaf']) or ((data[min_gini_feature] <= min_gini_feature_point).sum() < self.params['min_samples_leaf']):
                            subTree = data[y].value_counts().idxmax()

                        # 4. 小等于基尼指数阈值
                        elif min_gini < self.params['min_impurity_split']:
                            subTree = data[y].value_counts().idxmax()

                        # 5. 最大树深度
                        elif depth >= self.params['max_depth']:
                            subTree = data[y].value_counts().idxmax()

                    DTree[key] = subTree  # 该叶子结点的分裂特征
                else:
                    raise Exception("The subTree's value is not a DataFrame !")
            print("-- Leaf Node --")

        return DTree


    # 预测 ---------------------------------------------------------
    # 获取样本最多的类别
    def _most_leaf_node(self, tree, leaf_node_list):
        for value in tree.values():
            if isinstance(value, dict):
                self._most_leaf_node(value, leaf_node_list)
            else:
                leaf_node_list.append(value)
        return leaf_node_list  # return max(set(leaf_node_list), key=leaf_node_list.count)

    def _predict_one_by_one(self, DTree, row_data):
        for keys, values in DTree.items():
            T_key = keys
            T_value = values
            # print(T_key)
            # print(T_value)
            # print('---------------------------------------')

            T_key_list = re.split('(>|<=)', T_key)
            split_feature = T_key_list[0].strip()
            split_feature_oper = T_key_list[1].strip()
            split_feature_value = float(T_key_list[2].strip())

            # CART 二叉树
            if split_feature_oper == '>':
                if row_data[split_feature] > split_feature_value: # 符合就继续往下走
                    if isinstance(T_value, dict):  # 分支情况
                        return self._predict_one_by_one(DTree=T_value, row_data=row_data)
                    else:  # 叶子节点情况
                        return T_value
            if split_feature_oper == '<=':
                if row_data[split_feature] <= split_feature_value: # 符合就继续往下走
                    if isinstance(T_value, dict):  # 分支情况
                        return self._predict_one_by_one(DTree=T_value, row_data=row_data)
                    else:  # 叶子节点情况
                        return T_value


# --------------------------------- 测试 -------------------------------------- #
# 1.西瓜数据集
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

data = pd.read_csv('data/watermelon2.0.csv')
data = data.drop(['id'],axis=1)

# 增加连续型变量
data['density'] = [0.403, 0.556, 0.481, 0.666, 0.243, 0.437, 0.634, 0.556, 0.593, 0.774, 0.343, 0.639, 0.657, 0.666, 0.608, 0.719, 0.697]

# onehot
data, cates = one_hot_encoder(data=data,
                              categorical_features=['seze', 'gendi', 'qiaosheng', 'wenli', 'qibu', 'chugan'],
                              nan_as_category=False)

X = data.drop(['haogua'], axis=1)
y = data['haogua']

clf = CART(objective='binary')
clf.fit(X=X, y=y)
clf.DTree
y_test = clf.predict(new_data=X)

# # 2.Kaggle Titanic Data
# # 读取数据
# train = pd.read_csv('Data/train_fixed.csv')
# test = pd.read_csv('Data/test_fixed.csv')
#
# # onehot
# def one_hot_encoder(data, categorical_features, nan_as_category=True):
#     original_columns = list(data.columns)
#     data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
#     new_columns = [c for c in data.columns if c not in original_columns]
#     del original_columns
#     return data, new_columns
# train, cates = one_hot_encoder(data=train,
#                               categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
#                               nan_as_category=False)
# test, cates = one_hot_encoder(data=test,
#                               categorical_features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
#                               nan_as_category=False)
#
# # 分割数据
# train_train, train_test = train_test_split(train,test_size=0.4,random_state=0)
#
# X_train = train_train.drop(['Survived'], axis=1)
# y_train = train_train['Survived']
# X_test = train_test.drop(['Survived'], axis=1)
# y_test = train_test['Survived']
#
# # 分类器
# clf = CART(min_samples_leaf=100)
# # 训练
# start = time.clock()
# clf.fit(X=X_train, y=y_train)
# elapsed = (time.clock() - start)
# print("Train Model Time : ", elapsed)
# # 打印树
# clf.DTree
# # 预测
# start = time.clock()
# y_test_pred = clf.predict(new_data=X_test)
# elapsed = (time.clock() - start)
# print("Predict Model Time : ", elapsed)
#
# # AUC
# pre_dt = pd.DataFrame({'Y': train_test['Survived'],'pre_Y': y_test_pred})
# print('AUC for Test : ', roc_auc_score(pre_dt.Y,pre_dt.pre_Y))
#
# # Submit
# pre_Y = clf.predict(new_data=test)   # Parch = 9， 训练集未出现， 以该集合下最大类别代替
# submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
# submit.loc[:,'Survived'] = submit.loc[:,'Survived'].astype('category')
# submit['Survived'].cat.categories
# submit.to_csv('Result/submit_20181005.csv', index=False)
