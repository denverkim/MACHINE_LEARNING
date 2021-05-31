# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load the iris dataset and create a data frame named df (including target and feature names). 붓꽃데이터를 열어 타겟과 속성이름을 포함한 df 만들기
iris = datasets.load_iris()
iris
print(iris.DESCR)
iris.keys()
iris.data
iris.target
iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = iris.target
df

# Replace numeric category to string category (0  setosa, 1  versicolor, 2  virginica). 숫자로 된 범주를 글자로 바꾸기  
df['class'].replace(0,'setosa', inplace=True)
df['class'].replace(1,'versicolor', inplace=True)
df['class'].replace(2,'virginica', inplace=True)

# Summarize the data frame. 데이터프레임 요약 
# Shape, head(20), describe(), groupby(‘class’).size(), dtypes
df.shape
df.head(20)
df.describe()
tab = df.groupby('class').size()
pct = (tab / tab.sum())*100
tab = pd.concat([tab, pct], axis=1)
tab.columns = ['freq', 'percentage']
tab
plt.bar(tab.index, tab.freq)
plt.show()
sns.countplot(df['class'])

# Data visualization. 데이터시각화
# Univariate plot – boxplot, histogram. 박스플랏과 히스토그램을 이용한 일변량분석
df.hist()
plt.show()

fig, axes = plt.subplots(2,2)
sns.kdeplot(x='sepal_length', hue='class', data=df, ax=axes[0,0], legend=None)
sns.kdeplot(x='sepal_width', hue='class', data=df, ax=axes[0,1], legend=None)
sns.kdeplot(x='petal_length', hue='class', data=df, ax=axes[1,0], legend=None)
sns.kdeplot(x='petal_width', hue='class', data=df, ax=axes[1,1])
plt.tight_layout()

df.boxplot()
plt.show()
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
fig, axes = plt.subplots(ncols=4)
for i in range(4):
    sns.boxplot(data=df, y=df.columns[i], ax=axes[i])
plt.tight_layout()

fig, axes = plt.subplots(2,2)
sns.boxplot(x='class', y='sepal_length', data=df, ax=axes[0,0])
sns.boxplot(x='class', y='sepal_width', data=df, ax=axes[0,1])
sns.boxplot(x='class', y='petal_length', data=df, ax=axes[1,0])
sns.boxplot(x='class', y='petal_width', data=df, ax=axes[1,1])

# Multivariate plot – scatter_matrix. 산점도를 이용한 다변량 분석
pd.plotting.scatter_matrix(df)
sns.pairplot(df, hue='class')