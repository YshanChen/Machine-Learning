# -*- coding: utf8 -*-
import numpy as np
import pandas as pd

# Kaggle Titanic Data
train = pd.read_csv('E:/GitHub/Algorithms/Data/train.csv')
train['set'] = 'train'

test = pd.read_csv('E:/GitHub/Algorithms/Data/test.csv')
test['Survived'] = np.full((test.shape[0], 1), 1, dtype=int)
test = test[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
test['set'] = 'test'

data = pd.concat([train,test])
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

# 1
data.ix[:,'Survived'] = data.ix[:,'Survived'].astype('category')

# 2
data['Pclass'].value_counts()
data.ix[:,'Pclass'] = data.ix[:,'Pclass'].astype('category')

# 3
data['Sex'].value_counts()
data.Sex = np.where(data.Sex == "male",1,2)
data.ix[:,'Sex'] = data.ix[:,'Sex'].astype('category')

# 4
data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data.Age = np.where(data.Age <= 10,1,
                    np.where(data.Age <= 20,2,
                             np.where(data.Age <= 30,3,
                                      np.where(data.Age <= 40,4,
                                               np.where(data.Age <= 50,5,6)))))
data.ix[:,'Age'] = data.ix[:,'Age'].astype('category')

# 5
data['SibSp'].value_counts()
data.ix[:,'SibSp'] = data.ix[:,'SibSp'].astype('category')

# 6
any(pd.isnull(data['Parch']))
data['Parch'].value_counts()
data.ix[:,'Parch'] = data.ix[:,'Parch'].astype('category')

# 7
pd.isnull(['Fare'])
data['Fare'] = data['Fare'].fillna(np.mean(data['Fare']))

data.Fare.describe()

data.Fare = np.where(data.Fare <= 7,1,
                     np.where(data.Fare <= 15,2,
                              np.where(data.Fare <= 32,3,
                                       np.where(data.Fare <= 50,4,
                                                np.where(data.Fare <= 80,5,6)))))
data.ix[:,'Fare'] = data.ix[:,'Fare'].astype('category')

# 8
any(pd.isnull(data['Embarked']))
data['Embarked'] = data['Embarked'].fillna(data.Embarked.value_counts()[0])
data.Embarked = np.where(data.Embarked == 'S',1,
                         np.where(data.Embarked == 'C',2,3))
data.ix[:,'Embarked'] = data.ix[:,'Embarked'].astype('category')

# 划分
train = data[data['set']=='train']
train = train.drop(['set'],axis = 1)
test = data[data['set']=='test']
test = test.drop(['set','Survived'],axis = 1)

# 输出
train.to_csv('Data/train_fixed.csv',index = False)
test.to_csv('Data/test_fixed.csv',index = False)
