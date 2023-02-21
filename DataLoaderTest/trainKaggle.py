import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class TitanicDataSet(Dataset):
    def __init__(self, filePath):
        xy = np.loadtxt(filePath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[1:, [0,2-12]])
        self.y_data = torch.from_numpy(xy[1:, [1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
#
# x_data, y_data = TitanicDataSet('./titanic/train.csv')
# print(x_data)
# print("--------")
# print(y_data)


all_df = pd.read_csv(r'./titanic/train.csv',encoding="ISO-8859-1", low_memory=False)
print(all_df)
#把需要的列放进一个列表中，表示选中这些列, 拿到我们需要的数据集
cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data= all_df[cols].drop(['Name'], axis=1)
data.head()
# 将性别为female用0代替， male用1代替ß
dict_sex = {'female': 0, 'male': 1}
data['Sex'] = data['Sex'].map(dict_sex)

# 登船口也和性别处理方法一样
dict_embarked = {'S': 0, 'C': 1, 'Q': 2}
data['Embarked'] = data['Embarked'].map(dict_embarked)
#该方法可以计算数据中分别有多少个空值
print(data.isnull().sum())
#因为有很多的年龄为空值， 所以我们用这个方法可以用年龄的平均值填充空位
age_mean = data['Age'].mean()
data['Age'] = data['Age'].fillna(age_mean)
#填充fare
fare_mean = data['Fare'].mean()
data['Fare'] = data['Fare'].fillna(fare_mean)
#因为哪个登船口上船对生还率影响不大, 所以用1登船口填充
data['Embarked'] = data['Embarked'].fillna(1)

print(data)
