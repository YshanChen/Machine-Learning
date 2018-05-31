# -*- coding: utf-8 -*-
"""
感知机的 python 实现

Eason

2017/9/24

from http://blog.csdn.net/Artprog/article/details/61923910
"""

# 导入库
from numpy import *
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

# 定义一个函数makeLinearSeparableData以产生我们需要的线性可分的数据
def makeLinearSeparableData(weights, numLines):
    ''' (list, int) -> array

    Return a linear Separable data set. 
    Randomly generate numLines points on both sides of 
    the hyperplane weights * x = 0.

    Notice: weights and x are vectors.

    >>> data = pla.makeLinearSeparableData([2,3],5)
    >>> data
    array([[ 0.54686091,  3.60017244,  1.        ],
           [ 2.0201362 ,  7.5046425 ,  1.        ],
           [-3.14522458, -7.19333582, -1.        ],
           [ 9.72172678, -7.99611918, -1.        ],
           [ 9.68903615,  2.10184495,  1.        ]])
    >>> data = pla.makeLinearSeparableData([4,3,2],10)
    >>> data
    array([[ -4.74893955e+00,  -5.38593555e+00,   1.22988454e+00,   -1.00000000e+00],
           [  4.13768071e-01,  -2.64984892e+00,  -5.45073234e-03,   -1.00000000e+00],
           [ -2.17918583e+00,  -6.48560310e+00,  -3.96546373e+00,   -1.00000000e+00],
           [ -4.34244286e+00,   4.24327022e+00,  -5.32551053e+00,   -1.00000000e+00],
           [ -2.55826469e+00,   2.65490732e+00,  -6.38022703e+00,   -1.00000000e+00],
           [ -9.08136968e+00,   2.68875119e+00,  -9.09804786e+00,   -1.00000000e+00],
           [ -3.80332893e+00,   7.21070373e+00,  -8.70106682e+00,   -1.00000000e+00],
           [ -6.49790176e+00,  -2.34409845e+00,   4.69422613e+00,   -1.00000000e+00],
           [ -2.57471371e+00,  -4.64746879e+00,  -2.44909463e+00,   -1.00000000e+00],
           [ -5.80930468e+00,  -9.34624147e+00,   6.54159660e+00,   -1.00000000e+00]])
    '''
    w = array(weights)
    numFeatures = len(weights)
    dataSet = zeros((numLines, numFeatures + 1))
    
    # numlines 即样本个数
    for i in range(numLines):
        x = random.rand(1, numFeatures) * 20 - 10  #这个线性函数保证线性可分
        innerProduct = sum(w*x)
        if innerProduct <= 0:
            dataSet[i] = append(x,-1)
        else:
            dataSet[i] = append(x,1)
    
    return dataSet

'''
代码解释如下：
    weights 是一个列表，里面存储的是我们用来产生随机数据的那条直线的法向量。
    numLines 是一个正整数，表示需要创建多少个数据点。
    numFeatures 是一个正整数，代表特征的数量
    dataSet = zeros((numLines, numFeatures + 1)) 用于创建一个规模为numLines x (numFeatures + 1) 的数组，且内容全为 0。注意：numFeatures + 1 是为了在最后一列可以存储该数据点的分类（+1或者-1）。
    然后我们在 for循环里面填充 dataSet 的每一行。
    x = random.rand(1, numFeatures) * 20 - 10 产生一个数组，规模为一行，numFeatures 列， 每个数都是 -10 到 10 的随机数。
    innerProduct = sum(w * x) 计算内积
    接下来的 if 语句判断如果内积小于等于 0，则是负例，否则是正例
    numpy 提供的 append 函数可以扩充一维数组，可以自己实验一下。
    最后返回数据集合。
    函数的 docstring里面提供了使用例子，可以自己试一下，因为是随机数，所以结果不会相同。
'''
set_printoptions(threshold=np.inf) 
data = makeLinearSeparableData([2,3],100)

# 画图展示数据
def plotData(dataSet):
    ''' (array) -> figure

    Plot a figure of dataSet

    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = array(dataSet[:,2])
    idx_1 = where(dataSet[:,2]==1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==-1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
    plt.show()

plotData(data)

# 训练
def train(dataSet, plot = False):
    ''' (array, boolean) -> list

    Use dataSet to train a perceptron
    dataSet has at least 2 lines.

    '''
    
    numLines = dataSet.shape[0]             # 样本数
    numFeatures = dataSet.shape[1]          # 维度
    w = zeros((1, numFeatures - 1))         # 初始参数
    separated = False
    
    # 随机梯度下降
    i = 0                                   # 从第1轮迭代开始
    while not separated and i < numLines:   # 如果分类错误
        if dataSet[i][-1]*sum(w*dataSet[i,0:-1]) <= 0:   # 更新权重向量
            w = w + dataSet[i][-1]*dataSet[i,0:-1]    # 设置为未完全分开
            separated = False   # 重新开始遍历每个数据点
            i = 0
        else:
            i += 1   # 如果分类正确，检查下一个数据点
    
    if plot == True:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Linear separable data set')
        plt.xlabel('X')
        plt.ylabel('Y')
        labels = array(dataSet[:,2])
        idx_1 = where(dataSet[:,2]==1)
        p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], 
            marker='o', color='g', label=1, s=20)
        idx_2 = where(dataSet[:,2]==-1)
        p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], 
            marker='x', color='r', label=2, s=20)
        x = w[0][0] / abs(w[0][0]) * 10
        y = w[0][1] / abs(w[0][0]) * 10
        ann = ax.annotate(u"",xy=(x,y), 
            xytext=(0,0),size=20, arrowprops=dict(arrowstyle="-|>"))
        ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
        plt.legend(loc = 'upper right')
        plt.show()

    return w

'''
    代码解释：
    
    该函数有两个参数，地一个是数据集 dataSet，第二个是 plot，如果不提供值，有默认值 False，意思是只返回最后的结果，不绘制散点图。我这样设计这个训练函数，是为了方便查看训练完成后的结果。
    
    首先获得数据集的行数 numLines 和特征的数目 numFeatures，减一是因为最后一列是数据点的分类标签，分类标签并不是特征。
    
    创建一个数组 w 保存权重向量。
    
    while 循环只要满足任何一个条件就结束：已经完全将正例和负例分开，或者 i 的值超过样本的数量。其实第二个条件是不会发生的，因为感知机的训练算法是收敛的，所以一定会将数据完全分开，证明可见我的另一篇文章：感知机算法原理（PLA原理）及 Python 实现，但前提是数据集必须是线性可分的。
'''

# 检验算法
data = makeLinearSeparableData([99,1],100)
w = train(data, True)
w



    
    
    
    
    
    
    
    