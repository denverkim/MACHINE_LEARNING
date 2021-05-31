# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 08:54:22 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('E:/SEOULTECH/ML/LAB/')


# Import the train data of Loan Predication data set, do some basic exploration tasks, and use the graphs to visualize the data. 대출 훈련데이터를 가져와서 기본 탐색을 수행하고 시각화 하시오.
loan_train = pd.read_csv('loan_train.csv')
loan_train.head()
loan_train.shape
loan_train.dtypes
loan_train.describe()
loan_train.Gender.value_counts() 
col_names = loan_train.columns
for col in col_names:
    print(loan_train[col].value_counts())
loan_train.hist()
plt.tight_layout()
loan_train.boxplot()
plt.tight_layout()
loan_train.columns
sns.scatterplot(data=loan_train, x='ApplicantIncome', y='LoanAmount')

# Combine train and test data and fill missing data. 훈련과 테스트 데이터를 합친 후 분실값 처리
loan_test = pd.read_csv('loan_test.csv')
loan_test.shape
loan_train.shape
loan = pd.concat([loan_train, loan_test])
loan.shape
loan.isna().sum()

# Mean for Loan_Amount. 대출액은 평균으로 대체
loan.LoanAmount.fillna(loan.LoanAmount.mean(), inplace=True)

#Mode for Self_Employed, Gender, Married, Dependents, Loan_Amount_Term, and Credit_History. 나머지는 최빈수로 대체
# loan.Gender.mode()[0]
names = ['Self_Employed', 'Gender', 'Married', 'Dependents', 'Loan_Amount_Term', 'Credit_History']
for n in names:
    loan[n].fillna(loan[n].mode()[0], inplace=True)
    
# Encoding non-numeric values. 범주형 변수를 인코딩
# Use get_dummies for Property_Area
dummy = pd.get_dummies(loan.Property_Area)
loan = pd.concat([loan, dummy], axis=1)
loan.columns

# Use label encoder for Gender, Married, Education, Self_Employed (can use for loop)
loan_original = loan.copy()
bin_names = ['Gender', 'Married', 'Education', 'Self_Employed']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for b in bin_names:
    loan[b] = le.fit_transform(loan[b])
loan_original.head()
loan.head()

# Use replace for Loan_Status ({'Y':1,                        'N':0}, inplace=True) and Dependents ('3+','3', inplace=True) and change the data type to int32 using astype
loan.Loan_Status.replace({'Y':1, 'N':0}, inplace=True)
loan.dtypes
loan.Dependents.replace('3+', '3', inplace=True)
loan.Dependents = loan.Dependents.astype('int32')
loan.dtypes

# Divide dataset and x and y split. 데이터셋을 나눈 후 독립변수, 종속변수 분리
loan_train.shape
data = loan[:614]
test_data = loan[614:]
data.shape
test_data.shape

# Loan_Status for dependent variables.
y = data.Loan_Status

# Other variables (excluding 'Loan_ID', 'Loan_Status', 'Property_Area‘) for independent variables.
x = data.drop(['Loan_ID', 'Loan_Status', 'Property_Area'], axis=1)

# Train and test split. 훈련, 테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7) #데이터를 나눠서 80프로는 훈련용으로 20프로는 테스트용으로 사용함

# Build a logistic regression model. 예측모델 구축
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Check the accuracy score of the model. 성능평가
model.score(x_test, y_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

# Use 10-fold cross validation to evaluate the model. 교차검증을 사용하여 모델을 평가
from sklearn.model_selection import cross_val_score
cv = cross_val_score(model, x_test, y_test, cv=10, scoring='accuracy')
cv.mean(), cv.std()

# Create a confusion matrix, classification report, and ROC curve. 혼동행렬, 분류리포트, ROC커브
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_roc_curve(model, x_test, y_test)

# Predict with test data. 테스트 데이터를 가지고 예측
test_data.columns
test_data = test_data.drop(['Loan_ID', 'Property_Area','Loan_Status'], axis=1)
model.predict(test_data)

