# -*- coding: utf8 -*-
"""
Created on 2018/09/18
@author: Yshan.Chen

Commit：
1. 完成基尼指数函数；
2. 完成连续值的处理；
3. 完成生成树；

Todo List:
1. 缺失值的处理；
2. 树剪枝（前后）；
3. 预测；

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

    min_impurity_split: 节点划分最小不纯度，如果不纯度小等于该值则停止分裂。 △ Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
    max_features: 划分时考虑的最大特征数 △
    max_depth: 决策树最大深度 △
    min_samples_split: 内部节点再划分所需最小样本数
    min_samples_leaf: 叶子节点最少样本数 △
    min_weight_fraction_leaf: 叶子节点最小的样本权重和 (缺失值处理涉及)
    max_leaf_nodes: 最大叶子节点数
    """

    def __init__(self,
                 min_impurity_split=0.005,
                 max_features=2,
                 max_depth=5,
                 min_samples_leaf=2
                 ):
        self.params = {'min_impurity_split': min_impurity_split,
                       'max_features':max_features,
                       'max_depth':max_depth,
                       'min_samples_leaf':min_samples_leaf}
        self.DTree = {}

    def fit(self, X, y):
        self.DTree = self._fit(X=X, y=y)

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
            #     pre_y = most_leaf     # 出现NONE，强制赋值为初始样本样本数最多的类别！【待修改】
            predict_Y[row_index] = pre_y

        return predict_Y

    # 特征分裂向量
    '''
    # 计算每个特征的每个取值对应的Y类别的个数
    # 对于categorical和numeric特征都是二分法，categorical特征需要先onehot-encoding
    # 先排序，再去重，依次选择两个取值中分数进行二分,大于&小等于
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

            # 排序、去重
            feature_values_series = data[feature_name].sort_values().drop_duplicates(keep='first')

            # 特征的每个划分点对应的Y类别的个数
            feature_values_dict = {}
            for feature_value_1, feature_value_2 in zip(feature_values_series[0:], feature_values_series[1:]):
                feature_values_vec = [0, 0]  # [>a, <=a]

                # print(feature_value_1, feature_value_2)
                # 中位数作为候选划分数
                feature_value = round((feature_value_1+feature_value_2)/2, 4)

                Vec_bigger = np.zeros(shape=(1, y_class_num))[0]
                Vec_lesser = np.zeros(shape=(1, y_class_num))[0]

                for y_class_index, y_class in enumerate(y_classes):
                    count_number = ((data[feature_name] > feature_value) & (data[y] == y_class)).sum()
                    Vec_bigger[y_class_index] = count_number
                    feature_values_vec[0] = Vec_bigger

                for y_class_index, y_class in enumerate(y_classes):
                    count_number = ((data[feature_name] <= feature_value) & (data[y] == y_class)).sum()
                    Vec_lesser[y_class_index] = count_number
                    feature_values_vec[1] = Vec_lesser

                feature_values_dict[feature_value] = feature_values_vec

            # 打印:分裂特征 & 取值对应类别个数
            # print('Feature Split Name : ', feature_name)
            # print('Feature Class Number : ', feature_values_dict)

            feature_split_dic[feature_name] = feature_values_dict

        return feature_split_dic

    # 数据集D的基尼指数
    def _gini_D(self, Di_dic):
        # Di_dic => np.array
        if isinstance(Di_dic, dict):
            Di_dic = np.array(list(Di_dic.values()))

        # 总集合的个数
        D_num = Di_dic.sum()

        # 计算：数据集的gini
        g_vec = 0
        for C_k in Di_dic:
            g_vec = g_vec + (C_k/D_num)**2
        gini_D = 1 - g_vec

        return gini_D

    # 数据集在特征A下的条件下的基尼指数（选择基尼指数最小的作为最优特征及U最优特征点）
    def _gini_A(self, Di_dic, Aik_dic):
        # Di_dic => np.array
        if isinstance(Di_dic, dict):
            Di_dic = np.array(list(Di_dic.values()))

        # 总集合的个数
        D_num = Di_dic.sum()

        # 候选划分点
        A_toSelect = Aik_dic.keys()

        gini_A_dic = {}
        for a_i in A_toSelect:
            # 大于a_i的数据集
            D_bigger = Aik_dic[a_i][0]
            # 小等于a_i的数据集
            D_lesser = Aik_dic[a_i][1]

            # 数据集在特征A取该划分点的条件下的基尼指数
            # gini_set_bigger = self._gini_D(Di_dic=D_bigger)
            # gini_set_lesser = self._gini_D(Di_dic=D_lesser)
            gini_set_bigger = _gini_D(self=[], Di_dic=D_bigger)
            gini_set_lesser = _gini_D(self=[], Di_dic=D_lesser)
            gini_D_A = (D_bigger.sum() / D_num * gini_set_bigger) + (D_lesser.sum() / D_num * gini_set_lesser)

            gini_A_dic[a_i] = gini_D_A

        # 选取基尼指数最小的划分点为该特征的最优划分点，相应基尼指数为该特征的最优基尼指数
        gini_A_opt = [min(gini_A_dic, key=gini_A_dic.get), min(gini_A_dic.values())]

        return gini_A_opt

    # 计算每个特征的在每个划分点下的基尼指数，选取最小的基尼指数对应的特征以及最优划分点
    def _gini_min(self, data, y):
        # 特征变量个数
        X = data.drop([y], axis=1).columns
        X_number = len(X)

        # 每个特征的信息增益
        gain_dic = dict.fromkeys(X, 0)

        # 计算每个特征的每个取值对应的Y类别的个数
        # feature_split_dic = self._feature_split(data, y)
        feature_split_dic = _feature_split(self=[], data=data, y=y)

        # Y类别个数
        Di_dic = dict(data[y].value_counts())

        # 计算各特征的信息增益
        if gain_dic.keys() != feature_split_dic.keys():
            raise ValueError("features are wrong !")

        for feature_name in gain_dic.keys():  # feature_name = 'chugan_1'
            gain_dic[feature_name] = _gini_A(self=[], Di_dic=Di_dic, Aik_dic=feature_split_dic[feature_name])
            # gain_dic[feature_name] = self._gini_A(Di_dic=Di_dic, Aik_dic=feature_split_dic[feature_name])

        # 选取信息增益最大的特征
        min_gini = 2 # 基尼小等于1
        min_gini_feature = ''
        min_gini_feature_point = ''
        for feature, value in gain_dic.items():
            if value[1] < min_gini: # Todo:Or <= ?
                min_gini = value[1]
                min_gini_feature = feature
                min_gini_feature_point = value[0]

        # 返回 划分特征, 最优划分点, 最小基尼指数
        return [min_gini_feature, min_gini_feature_point, min_gini]

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
    # t = _growTree(self=[], X=X, y=y, DTree={})
    # t = _growTree(self=[], X=X_train, y=y_train, DTree={})
    def _growTree(self, X, y, DTree={}):
        # 初次分裂
        if DTree == {}:
            # Data
            data = pd.concat([X, y], axis=1).rename(str, columns={y.name: 'label'})
            data['label'] = data['label'].astype('category')

            # define y
            y = 'label'

            # 排除取值唯一的变量
            data = _drop_unique_column(self=[], data=data) #  Todo:data = self._drop_unique_column(data)

            # X
            X = data.drop([y], axis=1).columns

            # 生成树(干)
            DTree = {}

            # 计算划分特征，最优划分点，最小基尼指数
            gini_list = _gini_min(self=[], data=data, y=y)
            min_gini_feature = gini_list[0]
            min_gini_feature_point = gini_list[1]
            min_gini = gini_list[2]
            # gini_list = _gini_min(self=[], data=data, y=y)

            # 1. 如果不纯度<=阈值则停止分裂(min_impurity_split); 2. 类别取值<=1停止分裂； 3. 数据集为空停止分裂；
            if min_gini > 0.05 and len(data[y].unique()) > 1 and data.shape[0] > 0:       # Todo:self.params['min_impurity_split']
                splitting_feature = min_gini_feature
                splitting_point = min_gini_feature_point

                # 初次分裂
                print([splitting_feature, splitting_point])
                for opera in ['>', '<=']: # opera = '<='
                    if opera == '>': # 大于分裂点
                        data_split_temp = data[data[splitting_feature] > splitting_point]
                        # data_split_temp = self._drop_unique_column(data_split_temp) # 分裂后删除取值唯一的特征
                        data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                    else: # 小等于分裂点
                        data_split_temp = data[data[splitting_feature] <= splitting_point]
                        # data_split_temp = self._drop_unique_column(data_split_temp)  # 分裂后删除取值唯一的特征
                        data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                    description = ' '.join([str(splitting_feature), opera, str(splitting_point)])

                    # 继续分裂
                    if len(data_split_temp[y].unique()) > 1 and data_split_temp.shape[0] > 3 and data_split_temp.shape[1] > 1: # Todo:min_samples_leaf
                        currentTree = {description: data_split_temp}
                        sub_subTree = _growTree(self=[], X=X, y=y, DTree=currentTree) # sub_subTree = self._Decision_Tree(X=X, y=y, DTree=currentTree)
                        DTree.update(sub_subTree)

                    # 停止分裂
                    else:
                        # 如果分裂后类别唯一，则停止分裂，叶子结点为该类别
                        if len(data_split_temp[y].unique()) == 1:
                            currentValue = data_split_temp[y].unique()[0]

                        # 停止分裂判断：叶子结点的最小样本个数小于阈值，不再继续分裂，此分裂有效。叶子结点为样本最多的类别 (min_samples_leaf)
                        elif data_split_temp.shape[0] <= 3:  # min_samples_leaf
                            currentValue = data_split_temp[y].value_counts().idxmax()

                        # 停止分裂判断：没有可用于分裂的特征
                        elif data_split_temp.shape[1] <= 1:
                            currentValue = data_split_temp[y].value_counts().idxmax()

                        # 符合继续分裂条件
                        currentTree = {description: currentValue}
                        DTree.update(currentTree)

            # 不分裂
            else:
                if data.shape[0] <= 0: # 空集
                    print("Data set is None !")

                elif len(data[y].unique()) <= 1: # 类别值唯一
                    print("Y only one value !")

                elif min_gini <= 0.05: # 小等于基尼指数阈值 Todo: self.params['min_impurity_split']
                    print("initial min_gini <= min_impurity_split !")

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
                    data = value

                    # 特征变量X
                    X = data.drop([y], axis=1).columns

                    # 计算划分特征，最优划分点，最小基尼指数
                    # gini_list = self._gain_min(data, y)
                    gini_list = _gini_min(self=[], data=data, y=y)
                    min_gini_feature = gini_list[0]
                    min_gini_feature_point = gini_list[1]
                    min_gini = gini_list[2]

                    # 1. 如果不纯度<=阈值则停止分裂(min_impurity_split); 2. 类别取值<=1停止分裂；
                    if min_gini > 0.05 and len(data[y].unique()) > 1: # Todo: self.params['min_impurity_split'] = 0.05
                        splitting_feature = min_gini_feature
                        splitting_point = min_gini_feature_point

                        # 初次分裂
                        print([splitting_feature, splitting_point])

                        for opera in ['>', '<=']:
                            if opera == '>':  # 大于分裂点
                                data_split_temp = data[data[splitting_feature] > splitting_point]
                                # data_split_temp = self._drop_unique_column(data_split_temp) # 分裂后删除取值唯一的特征
                                data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                            else:  # 小等于分裂点
                                data_split_temp = data[data[splitting_feature] <= splitting_point]
                                # data_split_temp = self._drop_unique_column(data_split_temp)  # 分裂后删除取值唯一的特征
                                data_split_temp = _drop_unique_column(self=[], data=data_split_temp)
                            description = ' '.join([str(splitting_feature), opera, str(splitting_point)])

                            # 继续分裂-递归
                            if len(data_split_temp[y].unique()) > 1 and data_split_temp.shape[0] > 3 and data_split_temp.shape[1] > 1: # Todo: min_samples_leaf
                                currentTree = {description: data_split_temp}
                                sub_subTree = _growTree(self=[], X=X, y=y, DTree=currentTree)  # 递归   Todo: sub_subTree = self._CART(X=X, y=y, DTree=currentTree)
                                subTree.update(sub_subTree)

                            # 停止分裂判断，此分裂有效
                            else:
                                # 如果分裂后类别唯一，则停止分裂，叶子结点为该类别
                                if len(data_split_temp[y].unique()) == 1:
                                    currentValue = data_split_temp[y].unique()[0]

                                # 叶子结点的最小样本个数小于阈值，不再继续分裂。叶子结点为样本最多的类别 (min_samples_leaf)
                                elif data_split_temp.shape[0] <= 3:  # Todo:min_samples_leaf
                                    currentValue = data_split_temp[y].value_counts().idxmax()

                                # 没有可用于分裂的特征
                                elif data_split_temp.shape[1] <= 1:
                                    currentValue = data_split_temp[y].value_counts().idxmax()

                                # 符合继续分裂条件
                                currentTree = {description: currentValue}
                                subTree.update(currentTree)

                    # 停止分裂判断，此次不分裂
                    else:
                        if min_gini <= 0.05: # Todo: self.params['min_impurity_split'] = 0.05
                            subTree = data[y].value_counts().idxmax()

                        elif len(data[y].unique()) <= 1:
                            subTree = data[y].value_counts().idxmax()

                    DTree[key] = subTree

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

            T_key_list = re.split('(=|<|<=|>|>=|!=)', T_key)
            split_feature = T_key_list[0].strip()
            split_feature_oper = T_key_list[1].strip()
            split_feature_value = T_key_list[2].strip()

            # ID3 非二叉树
            if str(row_data[split_feature]) == split_feature_value:  # 符合就继续往下走
                if isinstance(T_value, dict):  # 还有分支情况
                    return self._predict_one_by_one(DTree=T_value, row_data=row_data)
                else:  # 叶子节点情况
                    return T_value

    # def predict(self, new_data):
    #     if self.DTree == {}:
    #         raise ValueError('There is no classifier for predicting !')
    #     else:
    #         return predict(DTree=self.DTree, new_data=new_data)


def main():

if __name__ == '__main__':
    main()


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


data = pd.concat([X, y], axis=1).rename(str, columns={y.name: 'label'})
y = 'label'


clf = DTree(method='C4.5', delta=0.01)
clf.fit(X=X, y=y)
clf.DTree
y_test = clf.predict(new_data=X)

# 2.Kaggle Titanic Data
# 读取数据
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

# 分类器
clf = DTree(method='C4.5', delta=0.01)
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
pre_Y = clf.predict(new_data=test.iloc[342:344,:])   # Parch = 9， 训练集未出现， 以该集合下最大类别代替
submit = pd.DataFrame({'PassengerId': np.arange(892,1310),'Survived': pre_Y})
submit.loc[:,'Survived'] = submit.loc[:,'Survived'].astype('category')
submit['Survived'].cat.categories
submit.to_csv('Result/submit_20180914.csv', index=False)
