#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:41:40 2018

@author: chenys
"""

from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[0,0],[1,1],[2,2]]
y = [0,1,2]

reg.fit (X, y)

reg.coef_
# array([1., 1.])
# array([0.5, 0.5])

# -------------------------
y = mat([[0],[1],[2]])

X = mat([[0,0],[1,1],[2,2]])
X_Inverse = X.I
X_T = X.T

pesudo_inverse_1 = X_T*X
pesudo_inverse_2 = pesudo_inverse_1.I   # singular matrix
pesudo_inverse_3 = pesudo_inverse_2*X_T
pesudo_inverse = pesudo_inverse_3

W = pesudo_inverse*y
#matrix([[1.],
#        [1.]])


# python calculate pesudo-inverse 
pesudo_inverse = np.linalg.pinv(X)

W = pesudo_inverse*y
#matrix([[1.],
#        [1.]])
#matrix([[0.5],
#        [0.5]])

HAT = X*pesudo_inverse
I = mat([[1,0,0],[0,1,0],[0,0,1]])
np.trace(I-HAT)
#2.0000000000000004    N-(d+1) = 3-(1+1) = 1 ?????????


# ---------------------------------------

import numpy as np
from numpy import *
a1=array([1,2,3]);
a1=mat(a1);


a1=mat([1,2]);      
a2=mat([[1],[2]]);
a3=a1*a2;
a4 = a2*a1
#1*2的矩阵乘以2*1的矩阵，得到1*1的矩阵

a1=mat([1,1]);
a2=mat([2,2]);
a3=multiply(a1,a2);

a1=mat(eye(2,2)*0.5);
a2=a1.I;
a3 = mat([[0,0],[1,1],[2,2]])
a4=a3.I;

#求矩阵matrix([[0.5,0],[0,0.5]])的逆矩阵

a1=mat([[1,1],[0,0]]);
a2=a1.T;




