# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:28:38 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('E:/SEOULTECH/ML/LAB/')

# Import the train and test data of Big Mart Sales and do some basic exploration tasks. 훈련, 테스트용 데이터를 임포트한 후에 기본 데이터탐색을 수행하시오.
# read_csv() 파일읽기
train = pd.read_csv('bigmart_train.csv')

# shape, dtypes, describe() 데이터검색 
train.shape
train.dtypes
a = train.describe(include='all')

# df.isnull().sum() 널값검색
train.isna().sum()

# 일변량 분석
# 계량형 – 히스토그램, 박스플랏
train.hist()
plt.tight_layout()
plt.show()
train.boxplot()
plt.tight_layout()
plt.show()

# 범주형 – 빈도테이블, 막대그래프
train.dtypes
tab = train.Item_Fat_Content.value_counts()
sns.barplot(x=tab.index, y=tab)
tab.plot(kind='bar')

# 이변량 분석
# 계량형 계량형 – 상관계수, 산점도
a = train.corr()
sns.heatmap(a, annot=True)
sns.pairplot(train)
from pandas.plotting import scatter_matrix
scatter_matrix(train)

# 범주형 계량형 - 막대그래프

# Set a value 1 for outlet sales in the test dataset. 테스트테이터에 아웃렛세일 열을 만들고 1을 할당
test = pd.read_csv('bigmart_test.csv')
test.shape
train.shape
train.columns
test['Item_Outlet_Sales'] = 1

# Combine train and test data. 훈련과 테스트데이터를 합침
df = pd.concat([train, test])
df.shape

# Impute the missing values and assign a value to mis-matched levels. 분실값 및 매칭되지 않는 값 대체 
# Impute missing values by median for item weight. 아이템 무게의 분실값을 중위수로 대체
df.isna().sum()
df.Item_Weight.fillna(df.Item_Weight.median(), inplace=True)

# Impute missing values by median for item visibility. 노출도의 분실값을 중위수로 대체
df.loc[df.Item_Visibility == 0,'Item_Visibility'] = df.Item_Visibility.median()
df.Item_Visibility.describe()

# Assign the  name “Other” to unnamed level in outlet size variable. 아웃렛사이즈의 분실값을 Other로 대체
df.columns
df.Outlet_Size.value_counts(dropna=False)
df.Outlet_Size.fillna('Other', inplace=True)

# Rename the various levels of item fat content. 지방함유율의 값 대체
# LF to Low Fat, reg to Regular, low fat to Low Fat
df.Item_Fat_Content.value_counts()
df.Item_Fat_Content.replace(['LF','low fat'], 'Low Fat', inplace=True)
df.Item_Fat_Content.replace('reg', 'Regular', inplace=True)

# Create a new column, Year =  2021 – outlet establishment year. 새로운 열 만들기
df['Year'] = 2021 - df.Outlet_Establishment_Year

# Drop the variables not required in prediction model (exclude item identifier, outlet identifier, item fat content, outlet establishment year, and item type). 필요없는 열을 삭제
df = df.drop(['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Outlet_Establishment_Year','Item_Type'], axis=1)
df.dtypes

# Encode the categorical variables. 범주형 변수를 인코딩
df_original  = df.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Outlet_Size = le.fit_transform(df.Outlet_Size)
df.Outlet_Location_Type = le.fit_transform(df.Outlet_Location_Type)
df.Outlet_Type = le.fit_transform(df.Outlet_Type)

# Divide the bigmart data frame into data and test_data. 데이터와 테스트 데이터로 나눔
train.shape
data = df[:8523]
test_data = df[8523:]

# X and y split
y = data.Item_Outlet_Sales
x = data.drop('Item_Outlet_Sales', axis=1)

# Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3, random_state=1)

# Regression and residual plot. 회귀분석과 잔차그래프
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test) #0.5027452145935561

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
r2_score(y_test, y_pred)

residual = y_test - y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)

# After log transformation, regression and residual plot. 로그변환 후 회귀과 잔차그래프
# X and y split
y = np.log(data.Item_Outlet_Sales)
x = data.drop('Item_Outlet_Sales', axis=1)

# Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3, random_state=1)

# Regression and residual plot. 회귀분석과 잔차그래프
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test) #0.5733699794773608

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
r2_score(y_test, y_pred)

residual = y_test - y_pred
std_residual = residual / np.std(residual)
sns.residplot(y_pred, std_residual)

# Check the r2 score and do some predictions. 성능평가 및 예측
test_data = test_data.drop('Item_Outlet_Sales', axis=1)
np.exp(model.predict(test_data))