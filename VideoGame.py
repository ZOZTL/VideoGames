#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

"""数据处理"""
# 数据加载
data = pd.read_csv("data/vgsales.csv", index_col=0)
# 缺失值处理
print("缺失数据为：")
print(data.isnull().sum())
# 删除缺失数据项
data = data.dropna(axis=0, subset=['Year', 'Publisher'])
print(data.isnull().sum())

print(data.info())

"""数据分析"""
# 各款电子游戏全球销售总额
# plt.figure(figsize=(25, 10))
# gameSale = data.groupby('Name', as_index=False).mean().sort_values(by='Global_Sales', ascending=False).head(10)
# sns.barplot(y='Global_Sales',
#             x='Name',
#             data=gameSale)
# plt.title('各款电子游戏全球销售总额')
# plt.show()

# 各类电子游戏类型全球销售总额
# plt.figure(figsize=(25, 10))
# genreSale = data.groupby('Genre', as_index=False).sum().sort_values(by='Global_Sales', ascending=False).head(20)
# sns.barplot(x='Genre',
#             y='Global_Sales',
#             data=genreSale)
# plt.title('各类电子游戏类型全球销售总额')
# plt.show()

# 各游戏平台全球销售总额
# plt.figure(figsize=(25, 10))
# platformSale = data.groupby('Platform', as_index=False).sum().sort_values(by='Global_Sales', ascending=False)
# sns.barplot(x='Platform',
#             y='Global_Sales',
#             data=platformSale)
# plt.title('各游戏平台全球销售额')
# plt.show()


# 各游戏发行人全球售额总额
# plt.figure(figsize=(25, 10))
# publisherSale = data.groupby('Publisher', as_index=False).sum().sort_values(by='Global_Sales', ascending=False).head(10)
# sns.barplot(x='Publisher',
#             y='Global_Sales',
#             data=publisherSale)
# plt.title('各游戏发行人全球售额总额')
# plt.show()

# 全球电子游戏销量走势
# label = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
# globalTrendSale = pd.pivot_table(data, index='Year', values=label, aggfunc=np.sum)
# fig = plt.figure(figsize=(15, 6))
# sns.lineplot(data=globalTrendSale,
#              hue="event",
#              style="event",
#              markers=True,
#              dashes=False)
# plt.title('全球电子游戏销量走势')
# plt.show()

"""预测"""
# 通过各市场销量预测全球销量
label = ['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales']
x = data[label]
y = data['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
SS = StandardScaler()
SS.fit(X_train)
X_train = SS.transform(X_train)
X_test = SS.transform(X_test)
LR = LinearRegression()
LR.fit(X_train, y_train)
LRTrain = LR.score(X_train, y_train)
LRTest = LR.score(X_test, y_test)
predictions = LR.predict(X_test)
print('训练值为:',LRTrain)
print('测试值为:',LRTest)
print('训练结果:', predictions)



