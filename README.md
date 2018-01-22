# Algorithms
## 简介
自己动手实现一些算法，顺便练习下Python。    
主要参考：
1. 李航《统计学习方法》
2. 周志华《机器学习》    

## 数据集
1. 数据集为Kaggle的Titanic数据集：https://www.kaggle.com/c/titanic/data
2. 直接调用处理后的数据集：_fixed.csv

## ID3
Score：0.74162    
Time：
* 训练：5.9002
* 预测：0.1884

一、问题：
1. 不剪枝情况下，对于新数据集会出现划分不到某一类的情况，输出None，手动划分为树模型叶子节点对应的最多的类别；
2. 如果某特征出现训练集中未包含的属性，则程序会报错；

## C4.5
一、相比ID3，存在以下改进：
1. 信息增益比；
2. 连续型特征的处理；
3. 缺失值的处理；
3. 剪枝；
