# Algorithms
## 简介
自己动手实现一些算法，顺便练习下Python。
纯当学习，谨慎参考:)

## 数据集
1. 数据集为Kaggle的Titanic数据集：https://www.kaggle.com/c/titanic/data
2. 直接调用处理后的数据集：_fixed.csv

## ID3
Score：0.74162    
Time：
* 训练：5.9002
* 预测：0.1884

问题：
1. 不剪枝情况下，对于新数据集会出现划分不到某一类的情况，输出None，手动划分为树模型叶子节点对应的最多的类别；
