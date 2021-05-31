# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:39:39 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('E:/SEOULTECH/ML/LAB/')

# Load College data (pd.read_csv). 데이터로링
college = pd.read_csv('College.csv')
college.columns

# Data exploration (head, describe, isna.sum(), set_index). 데이터탐색
college.head()
a = college.describe()
college.isna().sum()
college.set_index('Unnamed: 0', inplace=True)

# X and y split. 독립, 종속변수 분리
y = college.Private
x = college.drop('Private', axis=1)

# Data preprocessing (y - LabelEncoder, x - StandardScaler). 데이터전처리
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
y_original = y.copy()
le = LabelEncoder()
y = le.fit_transform(y)

x.boxplot()
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
plt.boxplot(x)

# Train and test split (sklearn.model_selection.train_test_split). 훈련, 테스트데이터 분리
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=7)

# Build a MLP model. MLP 모델 개발
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), random_state=0)
mlp.fit(x_train, y_train)
mlp.score(x_test,  y_test) #0.9230769230769231

# Evaluate the result. 결과평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
y_pred = mlp.predict(x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_roc_curve(mlp, x_test, y_test)
cross_val_score(mlp, x_test, y_test, cv=10, scoring='accuracy')

