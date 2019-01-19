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
<<<<<<< HEAD
    x_1 = data.drop(['y'], axis=1).reset_index(drop=True)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)}).reset_index(drop=True)
=======
    x_1 = data.drop(['y'], axis=1)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)})
>>>>>>> dd01e8b45af3171b6d17205db397fa5eeeca93bf
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 逆矩阵 & 转置矩阵
    X_T = X.T

    # 是否存在（X_T*X）的逆矩阵
    try:
        theta_opt = (X_T * X).I * X_T * y
    except:
        theta_opt = np.linalg.pinv(X_T * X) * X_T * y # pesudo_inverse

    return theta_opt

<<<<<<< HEAD
=======
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

>>>>>>> dd01e8b45af3171b6d17205db397fa5eeeca93bf
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
<<<<<<< HEAD
    x_1 = data.drop(['y'], axis=1).reset_index(drop=True)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)}).reset_index(drop=True)
=======
    x_1 = data.drop(['y'], axis=1)
    x_0 = pd.DataFrame({'x0': np.repeat(1, sample_number)})
>>>>>>> dd01e8b45af3171b6d17205db397fa5eeeca93bf
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
<<<<<<< HEAD
    x_1 = X.reset_index(drop=True)
    x_0 = pd.DataFrame({'x0': np.repeat(1, X.shape[0])}).reset_index(drop=True)
=======
    x_1 = X
    x_0 = pd.DataFrame({'x0': np.repeat(1, X.shape[0])})
>>>>>>> dd01e8b45af3171b6d17205db397fa5eeeca93bf
    X = np.mat(pd.concat([x_0, x_1], axis=1))

    # 计算
    y = X * theta

    return y

<<<<<<< HEAD
# 实际案例 -----------------------------------------------------------------
# 数据：http://archive.ics.uci.edu/ml/machine-learning-databases/00294/
# 里面是一个循环发电场的数据，共有9568个样本数据，每个数据有5列，分别是:AT（温度）, V（压力）, AP（湿度）, RH（压强）, PE（输出电力)。

# 1. sklearn ---------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mp
%matplotlib inline
mp.use('TkAgg')
from sklearn import datasets, linear_model

# read Data
data = pd.read_csv('/Users/chenys/Documents/Machine Learning/Algorithms/Algorithms/Linear Regression/data.csv')
X = data[['AT', 'V', 'AP', 'RH']]
X.head()
y = data[['PE']]
y.head()

# 划分训练集和测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 运行scikit-learn的线性模型
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# paras
print(linreg.intercept_) # [447.06297099]
print(linreg.coef_) # [[-1.97376045 -0.23229086  0.0693515  -0.15806957]]
# PE=447.06297099−1.97376045∗AT−0.23229086∗V+0.0693515∗AP−0.15806957∗RH　

# 模型评价
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred)) # MSE: 20.080401202073897
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE: 4.481116066570236

# 2. 正规方程 ---------------------------------------------------------
X_train.head()
y_train.columns = ['y']
train = pd.concat([y_train, X_train], axis=1)

X_test.head()
y_test.columns = ['y']
test = pd.concat([y_test, X_test], axis=1)

data = data[['PE','AT', 'V', 'AP', 'RH']]
data.columns = ['y','AT', 'V', 'AP', 'RH']

intercept_coef = normal_equation(data=train) 

#matrix([[ 4.47062971e+02],
#        [-1.97376045e+00],
#        [-2.32290859e-01],
#        [ 6.93514994e-02],
#        [-1.58069568e-01]])
# 一致 :)

y_pred = predict(X = X_test, theta = intercept_coef)

print("MSE:",metrics.mean_squared_error(y_test, y_pred)) # MSE: 20.080401202073897 VS 20.08040120234207
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE: 4.481116066570236 VS RMSE: 4.4811160666001575

# Plot
from ggplot import *
data = pd.read_csv('/Users/chenys/Documents/Machine Learning/Algorithms/Algorithms/Linear Regression/data.csv')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
Y_pred = pd.DataFrame(predict(X = X, theta = intercept_coef))
Y_pred.columns = ['PE']
df = pd.concat([y, Y_pred], axis=1)

pic = ggplot(aes(x = 'y', y = 'Y_pred'), data = df)
pic + geom_point()

# 3. 梯度下降 ---------------------------------------------------------

theta_gradient = gradient_descent(data=train)

#matrix([[4.54218423e+01],
#        [8.83179245e+02],
#        [2.45222007e+03],
#        [4.60261172e+04],
#        [3.33543833e+03]])
# 不一致 :(

# 预测
y_pred = predict(X = X_test, theta = theta_gradient)

print("MSE:",metrics.mean_squared_error(y_test, y_pred)) # MSE: 20.080401202073897 VS 2212899194773287
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE: 4.481116066570236 VS RMSE: 47041462.50674279

# Plot
data = pd.read_csv('/Users/chenys/Documents/Machine Learning/Algorithms/Algorithms/Linear Regression/data.csv')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
Y_pred = pd.DataFrame(predict(X = X, theta = theta_gradient))
Y_pred.columns = ['PE']
df = pd.concat([y, Y_pred], axis=1)

pic = ggplot(aes(x = 'y', y = 'Y_pred'), data = df)
pic + geom_point()







=======
# 最优参数
theta_opt = gradient_descent(data=df_1)

y = predict(X = pd.DataFrame({'x1':[1,2,3,4,5],'x2':[1,3,5,7,9]}))
>>>>>>> dd01e8b45af3171b6d17205db397fa5eeeca93bf
