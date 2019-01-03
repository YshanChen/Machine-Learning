# -*- coding: utf8 -*-
"""
Created on 2018/09/18
@author: Yshan.Chen

Update: 2018/10/23
Update: 2018/12/27

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
    min_samples_leaf: 叶子节点最少样本数 △ 如分裂后叶子节点样本数小于该值，则不分裂。
    min_weight_fraction_leaf: 叶子节点最小的样本权重和 (缺失值处理涉及)
    max_leaf_nodes: 最大叶子节点数
    """

    def __init__(self, objective,
                 max_features=0,
                 max_depth=3,
                 min_samples_split=5,
                 min_samples_leaf=1,
                 min_impurity_split=0.005):
        if objective not in ["regression", "binary"]:
            raise Exception("Error: objective must be \"regression\" or \"binary\" !")
        else:
            self.params = {'objective': objective, 'min_impurity_split': min_impurity_split,
                           'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf}
            self.DTree = {}

    def fit(self, X, y):
        if self.params['objective'] == 'binary':
            self.DTree = self._Decision_Tree_binary(X=X, y=y)
        if self.params['objective'] == 'regression':
            self.DTree = self._Decision_Tree_regression(X=X, y=y)

    def predict(self, new_data):  # 逐条预测，未实现并行化
        if self.DTree == {}:
            raise ValueError('There is no DecisionTree for predicting !')

        predict_Y = pd.Series([])

        # 样本最多的类别
        if self.params['objective'] == 'binary':
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

    # 排除取值唯一的变量
    def _drop_unique_column(self, data):
        del_unique_columns = []
        for col in [x for x in data.columns if x != 'label']:
            if len(data[col].unique()) <= 1:
                del_unique_columns.append(col)
        data = data.drop(del_unique_columns, axis=1)
        return data

    # 特征分裂，确定：(最优分裂特征，最优分裂点，最小平方损失)
    def _feature_split(self, data):  # data=data; y=y
        feature_split_dic = {}

        # X个数
        X = data.drop(['label'], axis=1).columns
        X_num = len(X)

        # 遍历特征，计算：每个特征的最小平方损失
        for feature_name in X:  # feature_name = 'crim'

            # 固定特征，遍历所有切分点，计算：每个切分点下的c1 and c2, c=avg(y)
            # 排序、去重、取所有特征值
            feature_values_series = data[feature_name].sort_values().drop_duplicates(keep='first')

            feature_values_dict = {}
            for feature_value_1, feature_value_2 in zip(feature_values_series[0:], feature_values_series[1:]): # feature_value_1=0.00632; feature_value_2=0.00906
                feature_values_vec = {'<= a': 0, '> a': 0, 'Square loss': 0}  # [>a, <=a]
                feature_split_value = round((feature_value_1 + feature_value_2) / 2, 4)

                y1 = data.loc[data[feature_name] <= feature_split_value, 'label']  # <=a
                y2 = data.loc[data[feature_name] > feature_split_value, 'label']   # >a
                c1 = y1.mean()
                c2 = y2.mean()
                square_loss = sum((y1 - c1)**2) + sum((y2 - c2)**2)  # 计算损失函数 min( min(sum(y1-c1)**2) + min(sum(y2-c2)**2) ); min(sum(y1-c1)**2) => c1_hat=mean(y1)

                feature_values_vec['<= a'] = c1
                feature_values_vec['> a'] = c2
                feature_values_vec['Square loss'] = square_loss

                feature_values_dict[feature_split_value] = feature_values_vec

            # 筛选：平方损失最小的切分点，作为该特征的最优切分点及最小损失
            feature_point = 0
            feature_squareloss_min = 1e10
            for a in feature_values_dict:
                # print('---------------')
                # print(a)
                # print(feature_values_dict[a])
                feature_squareloss = feature_values_dict[a]['Square loss']
                if feature_squareloss<feature_squareloss_min:
                    feature_squareloss_min = feature_squareloss
                    feature_point = a

            # 输出：(最优切分点，最小平方损失)
            feature_split_list = (feature_point, feature_squareloss_min)

            # 输出：每个特征的(最优切分点，最小平方损失)
            feature_split_dic[feature_name] = feature_split_list

        # 筛选：平方损失最小的特征，作为分裂特征(最优分裂特征，最优分裂点，最小平方损失)
        split_feature = '' # 最优分裂特征
        split_feature_point = 0 # 最优分裂点
        split_squareloss_min = 1e10 # 最小平方损失
        for feature_name in X:
            # print('--------------')
            # print(feature_name)
            # print(feature_split_dic[feature_name])
            if feature_split_dic[feature_name][1]<=split_squareloss_min:     # 'crim': (9.2807, 23018.036051194547)
                split_feature = feature_name
                split_feature_point = feature_split_dic[feature_name][0]
                split_squareloss_min = feature_split_dic[feature_name][1]

        # 输出：(最优分裂特征，最优分裂点，最小平方损失)
        split_list = (split_feature, split_feature_point, split_squareloss_min)

        return split_list

    # 训练 ---------------------------------------------------------
    """
    树的生成。
    - 分裂后特征是否还能继续用于分裂问题，取决于分裂后是否取值唯一，是否还有区分能力。
      对于离散型且onehot处理过的分裂特征，分裂后其特征取值唯一，故能够删除；
      对于连续型分裂特征，分裂后依然可能取值不唯一，故可能保留用于继续分裂；
      非分裂特征随着分裂可能也出现取值唯一情况，故每次分裂后均根据区分能力删除取值唯一的特征；
    """

    def _Decision_Tree_regression(self, X, y, DTree={}, depth=0):  # X=train_X; y=train_Y; DTree={}; depth=0
        # 初次分裂
        if DTree == {}:

            # Data
            data = pd.concat([X, y], axis=1).rename(str, columns={y.name: 'label'})
            data = self._drop_unique_column(data=data)  # 排除取值唯一的变量
            # data = _drop_unique_column(self=[], data=data)
            X = data.drop(['label'], axis=1).columns

            # 生成树桩
            DTree = {}
            depth = 0

            # 计算: (平方损失函数) 划分特征，划分点，最小平方损失
            split_list = self._feature_split(data=data)
            # split_list = _feature_split(self=[], data=data)

            '''
            分裂判断：
            1. 用于分裂结点的样本数小于min_samples_split,不分裂；
            2. 分裂后的两个结点的样本个数小于min_samples_leaf,不分裂；
            3. 树深度>max_depth,不分裂；
            '''
            if (data.shape[0] > self.params['min_samples_split']) & \
                    ((data[split_list[0]] <= split_list[1]).sum() > self.params['min_samples_leaf']) & \
                    ((data[split_list[0]] > split_list[1]).sum() > self.params['min_samples_leaf']) & \
                    (depth <= self.params['max_depth']):

                # 确定分裂 ---
                split_feature = split_list[0]
                split_feature_point = split_list[1]
                depth = depth + 1
                print(split_list)

                # 分裂ing ---
                for opera in ['<=', '>']:  # 分别处理左右两个branch  opera = '>'
                    if opera == '>':  # 大于分裂点
                        data_split_temp = data[data[split_feature] > split_feature_point]
                    else:  # 小等于分裂点
                        data_split_temp = data[data[split_feature] <= split_feature_point]

                    data_split_temp = self._drop_unique_column(data=data_split_temp)  # 分裂后删除取值唯一的特征
                    # data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                    description = ' '.join([str(split_feature), opera, str(split_feature_point)])

                    # 对于分裂后的结点的处理（判断是否满足叶子结点的条件，是否还要进行分裂）
                    # if len(data_split_temp[y].unique()) == 1:  # 1. 如果分裂后类别唯一，则停止分裂。结点为叶子结点，该类别即为输出。
                    #     currentValue = data_split_temp[y].value_counts().idxmax()
                    #     currentTree = {description: currentValue}
                    #     DTree.update(currentTree)

                    # if data_split_temp.shape[1] <= self.params[
                    #     'max_features'] + 1:  # 2. 停止分裂判断：可用于分裂的特征小于最大特征阈值。最大类别即为输出。
                    #     currentValue = data_split_temp[y].value_counts().idxmax()
                    #     currentTree = {description: currentValue}
                    #     DTree.update(currentTree)

                    # 分裂后结点非叶子结点，继续分裂
                    currentTree = {description: data_split_temp}
                    sub_subTree = self._Decision_Tree_regression(X=X, y='label', DTree=currentTree, depth=depth)
                    # X=X; y='label'; DTree=currentTree; depth=depth
                    DTree.update(sub_subTree)

            # 确定不分裂 -------
            else:
                # 1. 内部结点样本数小等于最小划分样本数阈值
                if data.shape[0] <= self.params['min_samples_split']:
                    print("split_sample <= min_samples_split !")

                # 2. 分裂后叶子结点样本数小等于min_samples_leaf, 不分裂
                elif ((data[split_list[0]] <= split_list[1]).sum() <= self.params['min_samples_leaf']) or \
                        ((data[split_list[0]] > split_list[1]).sum() <= self.params['min_samples_leaf']):
                    print("samples_leaf <= min_samples_leaf !")

                # 3. 最大树深度
                elif depth > self.params['max_depth']:
                    print("depth > max_depth !")

        # 第二次及之后的分裂
        else:
            key = list(DTree.keys())[0]  # key = 'rm > 6.945' key = 'rm > 7.445'
            value = DTree[key]  # value = DTree[key]

            # 子树
            subTree = {}

            # 判断是否为叶子结点
            if isinstance(value, pd.DataFrame):
                data = value  # 子集作为新的data

                # 特征变量X
                X = data.drop(['label'], axis=1).columns

                # 计算: (平方损失函数) 划分特征，划分点，最小平方损失
                split_list = self._feature_split(data=data)
                # split_list = _feature_split(self=[], data=data)

                '''
                分裂判断：
                1. 用于分裂结点的样本数小于min_samples_split,不分裂；
                2. 分裂后的两个结点的样本个数小于min_samples_leaf,不分裂；
                3. 树深度>max_depth,不分裂；
                '''
                if (data.shape[0] > self.params['min_samples_split']) & \
                        ((data[split_list[0]] <= split_list[1]).sum() > self.params['min_samples_leaf']) & \
                        ((data[split_list[0]] > split_list[1]).sum() > self.params['min_samples_leaf']) & \
                        (depth <= self.params['max_depth']):

                    # 确定分裂 ---
                    split_feature = split_list[0]
                    split_feature_point = split_list[1]
                    depth = depth + 1
                    print(split_list)

                    for opera in ['<=', '>']:  # 分别处理左右两个branch  opera = '>'
                        if opera == '>':  # 大于分裂点
                            data_split_temp = data[data[split_feature] > split_feature_point]
                        else:  # 小等于分裂点
                            data_split_temp = data[data[split_feature] <= split_feature_point]

                        data_split_temp = self._drop_unique_column(data=data_split_temp)  # 分裂后删除取值唯一的特征
                        # data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                        description = ' '.join([str(split_feature), opera, str(split_feature_point)])

                        # 对于分裂后的结点的处理（判断是否满足叶子结点的条件，是否还要进行分裂）
                        # if len(data_split_temp[y].unique()) == 1:  # 1. 如果分裂后类别唯一，则停止分裂。结点为叶子结点，该类别即为输出。
                        #     currentValue = data_split_temp[y].value_counts().idxmax()
                        #     currentTree = {description: currentValue}
                        #     subTree.update(currentTree)

                        if data_split_temp.shape[1] <= self.params[
                            'max_features'] + 1:  # 1. 停止分裂判断：可用于分裂的特征小于最大特征阈值。c = avg(y)
                            currentValue = data_split_temp['label'].mean()
                            currentTree = {description: currentValue}
                            subTree.update(currentTree)

                        elif depth >= self.params['max_depth']:  # 2. 树深度达到最大深度，则停止分裂: c = avg(y)
                            currentValue = data_split_temp['label'].mean()
                            currentTree = {description: currentValue}
                            subTree.update(currentTree)

                        else:  # 分裂后结点非叶子结点，继续分裂
                            currentTree = {description: data_split_temp}
                            sub_subTree = self._Decision_Tree_regression(X=X, y='label', DTree=currentTree, depth=depth)
                            # X=X; y='label'; DTree=currentTree; depth=depth
                            subTree.update(sub_subTree)

                # 确定不分裂 结点作为叶子结点 -------
                else:
                    # 1. 内部结点样本数小等于最小划分样本数阈值
                    if data.shape[0] <= self.params['min_samples_split']:
                        subTree = data['label'].mean()

                    # 2. 分裂后叶子结点样本数少于min_samples_leaf, 不分裂
                    elif ((data[split_list[0]] <= split_list[1]).sum() <= self.params['min_samples_leaf']) or \
                            ((data[split_list[0]] > split_list[1]).sum() <= self.params['min_samples_leaf']):
                        subTree = data['label'].mean()

                    # 3. 最大树深度
                    elif depth > self.params['max_depth']:
                        subTree = data['label'].mean()

                DTree[key] = subTree  # 该叶子结点的分裂特征
            else:
                raise Exception("The subTree's value is not a DataFrame !")  # print("-- Leaf Node --")

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
                if row_data[split_feature] > split_feature_value:  # 符合就继续往下走
                    if isinstance(T_value, dict):  # 分支情况
                        return self._predict_one_by_one(DTree=T_value, row_data=row_data)
                    else:  # 叶子节点情况
                        return T_value
            if split_feature_oper == '<=':
                if row_data[split_feature] <= split_feature_value:  # 符合就继续往下走
                    if isinstance(T_value, dict):  # 分支情况
                        return self._predict_one_by_one(DTree=T_value, row_data=row_data)
                    else:  # 叶子节点情况
                        return T_value


# # --------------------------------- 测试 -------------------------------------- #
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# 1. Boston Housing
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
submission['medv'] = test_y_pred
submission.to_csv('Result/Boston_Housing_190104.csv', index=False)

# crim
# per capita crime rate by town.
#
# zn
# proportion of residential land zoned for lots over 25,000 sq.ft.
#
# indus
# proportion of non-retail business acres per town.
#
# chas
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#
# nox
# nitrogen oxides concentration (parts per 10 million).
#
# rm
# average number of rooms per dwelling.
#
# age
# proportion of owner-occupied units built prior to 1940.
#
# dis
# weighted mean of distances to five Boston employment centres.
#
# rad
# index of accessibility to radial highways.
#
# tax
# full-value property-tax rate per $10,000.
#
# ptratio
# pupil-teacher ratio by town.
#
# black
# 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
#
# lstat
# lower status of the population (percent).
#
# medv
# median value of owner-occupied homes in $1000s.