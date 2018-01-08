# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from math import log
from treelib import *
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from sklearn import preprocessing

# ------------------------- 数据集 --------------------------- #
df = pd.read_csv('Data/ID3.csv',encoding="GBK")
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

# ---------------------------- 树类 -------------------------------------------------- #
DTree = Tree()
>>> DTree.create_node("Harry", "harry",data=df)  # root node
>>> DTree.create_node("Jane", "jane", parent="harry")
>>> DTree.create_node("Bill", "bill", parent="harry")
>>> DTree.create_node("Diane", "diane", parent="jane")
>>> DTree.create_node("Mary", "mary", parent="diane")
>>> DTree.create_node("Mark", "mark", parent="jane")
>>> DTree.show()
Harry
├── Bill
└── Jane
    ├── Diane
    │   └── Mary
    └── Mark




P = DTree.root
DTree.all_nodes()
DTree.children(DTree.subtree('diane').root)
Node.bpointer()

>>> sub_t = DTree.subtree('diane')
>>> sub_t.show()
Diane
└── Mary

DTree.all_nodes()

>>> DTree.remove_node(1)
>>> DTree.show()
Harry
├── Bill
└── Jane
    ├── Diane
    │   └── Mary
    └── Mark

# ----------------------------------  信息增益算法 ------------------------------------- #
# Di_vec = np.array([6,9])
# A1k_vec = np.array([[3,2],[2,3],[1,4]])
# A2k_vec = np.array([[6,4],[0,5]])
# A3k_vec = np.array([[6,3],[0,6]])
# A4k_vec = np.array([[4,1],[2,4],[0,4]])


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

    return  feature_split_num


# 数据集D的经验熵
def entropy(Di_vec):
    D = Di_vec.sum()
    if D==0:
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
# 输入：训练集D，特征集A，阈值\delta；
#
# 输出：决策树T；
#
# 1）若D中所有实例属于同一类C_k，则T为单结点树，并将类C_k作为该结点的类标记，返回T；
#
# 2）若特征集A=\emptyset，则T为单结点树，并将D中实例数最大的类C_k作为该结点的类标记，返回T；
#
# 3）否则，按信息增益算法计算A中各特征对D的信息增益，选择信息增益最大的特征A_g;
#
# 4）如果A_g的信息增益小于阈值\delta，则置T为单结点树，并将D中实例数最大的类C_k作为该结点的类标记，返回T；
#
# 5）否则，对A_g的每一个可能的值a_i，依A_g=a_i将D分割为若干个非空子集D_i，将D_i中实例数最大的类作为标记，构建子结点，由结点及其子结点构成数T，返回T；
#
# 6）对第i个子结点，以D_i为训练集，以A-\{A_g\}（该分支用过的特征不能再用）为特征集，递归地调用步(1)~步(5)，得到子树T_i，返回T_i；

y = 'class'
df = df

def ID3(df, y = 'class', delta = 0.005):
    currentTree = Tree()

    df_list = [[df,'','']]
    splitFearture = []
    node_no = 0
    parnode_no = 0

    # D中实例最大的类
    max_class_in_D = df[y].value_counts().argmax()

    # df_sub_group = df_list[0]
    for df_sub_group in df_list:
        df_sub = df_sub_group[0]
        par = df_sub_group[1]
        description = df_sub_group[2]

        if (len(df_sub[y].unique()) != 1) & (df_sub.empty != True):
            node_no = node_no + 1

            # 若信息增益大于阈值，则写入分裂特征
            if gain_max(df_sub,y)[1] > delta:
                split_feature_name = gain_max(df_sub,y)[0]
                if df_sub.equals(df_list[0][0]):
                    currentTree.create_node(str(split_feature_name),node_no,data=df_sub)
                else:
                    currentTree.create_node(''.join([str(split_feature_name),'(',description,')']),node_no,parent=par,
                                            data=df_sub)

                # 分裂
                for cat in np.unique(df_sub[split_feature_name]):
                    df_split_temp = df_sub[df_sub[split_feature_name] == cat].drop(split_feature_name,axis=1)
                    description = ' '.join([str(split_feature_name),'=',str(cat)])
                    df_list.append([df_split_temp,node_no,description])
            else:
                leaf_node = max_class_in_D
                if df_sub.equals(df_list[0][0]):
                    currentTree.create_node(''.join([str(leaf_node),'(',description,')']),node_no,
                                            data=df_sub)
                else:
                    currentTree.create_node(''.join([str(leaf_node),'(',description,')']),node_no,parent=par,
                                            data=df_sub)

        elif (len(df_sub[y].unique()) == 1):
            node_no = node_no + 1
            leaf_node = df_sub[y].values[0]
            currentTree.create_node(''.join([str(leaf_node),'(',description,')']),node_no,parent=par,data=df_sub)

        elif df.empty == True:
            node_no = node_no + 1
            leaf_node = max_class_in_D
            currentTree.create_node(''.join([str(leaf_node),'(',description,')']),node_no,parent=par,data=df_sub)

    return currentTree

DTree = ID3(df,'class')
DTree.show()


















