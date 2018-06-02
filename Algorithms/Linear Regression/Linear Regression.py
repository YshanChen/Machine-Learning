#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:41:40 2018

@author: chenys
"""
# 1. sklearn
# 2. 正规方程
# 3. 梯度下降法

from sklearn import linear_model
import numpy as np
import scipy as sp
import pandas as pd

# Data ------------------------------------------------
# 线性
df_1 = pd.DataFrame({'y' : [0,3,6], 'x1':[0,1,2], 'x2' : [0,1,2]},columns=['y','x1','x2'])
df_2 = pd.DataFrame({'y' : [0,3,6], 'x0':[1,1,1], 'x1':[0,1,2], 'x2' : [0,1,2]},columns=['y','x0','x1','x2'])

# 1. sklearn 实现 Linear Reg ---------------------------------------------
reg = linear_model.LinearRegression()

reg.fit(X=df_2[['x0','x1','x2']],y=df.y)

reg.coef_
reg.intercept_

# 2. 正规方程  ---------------------------------------------
# 1）不需要归一化特征
# 2）不适合特征数多的情况，计算量O(n^3), 千位级OK

def normal_equation(data):
    # 矩阵转换
    sample_number = data.shape[0]
    theta_number = data.shape[1] - 1 + 1

    y = np.mat(data['y']).T
    x_1 = data.drop(['y'], axis=1)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)})
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 逆矩阵 & 转置矩阵
    X_T = X.T

    # 是否存在（X_T*X）的逆矩阵
    try:
        theta_opt = (X_T * X).I * X_T * y
    except:
        theta_opt = np.linalg.pinv(X_T * X) * X_T * y # pesudo_inverse

    return theta_opt

# 预测 (不加截距项)
def predict(X, theta = theta_opt):
    # 矩阵转换
    x_1 = X
    x_0 = pd.DataFrame({'x0': np.repeat(1, X.shape[0])})
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 计算
    y = X * theta

    return y

# 最优参数
theta_opt = normal_equation(df_1)

y = predict(X = pd.DataFrame({'x1':[1,2,3,4,5],'x2':[1,3,5,7,9]}))

# 3. 梯度下降法 (矩阵形式) (不需要归一化特征) --------------------------------
# 1）需要归一化特征
# 2）适合特征数多的情况

# 损失函数
def cost_function(theta, X, y):
    cost_function = np.array(sum(np.power(X*theta-y,2)))
    return cost_function

# 梯度下降
def gradient_descent(data, eta = 0.1, delta = 0.000001):
    # 矩阵转换
    sample_number = data.shape[0]
    theta_number = data.shape[1] - 1 + 1

    y = np.mat(data['y']).T
    x_1 = data.drop(['y'], axis=1)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)})
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 参数初始化
    theta = np.mat(np.repeat(0, theta_number)).T
    theta_number = X.shape[1]
    theta_next = np.mat(np.repeat(np.nan, theta_number)).T

    while True:
        for theta_i in np.arange(0, theta_number):
            theta_next[theta_i] = theta[theta_i] - eta * (1/sample_number) * (X[:,theta_i].T * (X * theta - y))

        if (cost_function(theta, X, y) - cost_function(theta_next, X, y)<delta):
            theta_opt = theta_next
            return theta_opt
            break
        else:
            theta = theta_next
            theta_next = np.mat(np.repeat(np.nan, theta_number)).T

# 预测 (不加截距项)
def predict(X, theta = theta_opt):
    # 矩阵转换
    x_1 = X
    x_0 = pd.DataFrame({'x0': np.repeat(1, X.shape[0])})
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 计算
    y = X * theta

    return y

# 最优参数
theta_opt = gradient_descent(data=df_1)

y = predict(X = pd.DataFrame({'x1':[1,2,3,4,5],'x2':[1,3,5,7,9]}))