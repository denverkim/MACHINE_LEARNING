# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:07:21 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('E:/SEOULTECH/ML/LAB/')
tennis = pd.read_csv('PlayTennis.csv')
tennis.head()
tennis.dtypes

#encoding
tennis_original = tennis.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in tennis.columns:
    tennis[col] = le.fit_transform(tennis[col])

#x and y split
y = tennis['Play Tennis']
x = tennis.drop('Play Tennis', axis=1)    

#train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x,y,test_size=.3, random_state=1)

# Build and evaluate a naïve Bayesian model using PlayTennis.csv. 테니스데이터를 이용해서 나이브베이즈 모델 생성
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_score = gnb.score(x_test, y_test) #.8

from sklearn.metrics import accuracy_score
y_pred = gnb.predict(x_test)
accuracy_score(y_test, y_pred)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
bnb_score = bnb.score(x_test, y_test) #.6

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_score = mnb.score(x_test, y_test) #.8

# Compare the results. 결과 비교
print(gnb_score, bnb_score, mnb_score)

# Prediction with test data. 테스트 데이터로 예측
x_train[:1]
test_data = pd.DataFrame([[1,0,1,1]], columns=x_train.columns)
gnb.predict(test_data)
