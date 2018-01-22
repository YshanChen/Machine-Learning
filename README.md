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
Score：0.74641
Time：
* 训练：6.6842
* 预测：0.1371

## C4.5
一、相比ID3，存在以下改进：
1. 信息增益比；
2. 连续型特征的处理；
3. 缺失值的处理；
3. 剪枝；
