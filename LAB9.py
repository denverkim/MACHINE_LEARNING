# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:28:53 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer
import os
os.chdir('E:/SEOULTECH/ML/LAB/')

pima = pd.read_csv('pima-indians-diabetes.csv', header=None)
pima.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

y = pima['class']
x = pima.drop('class', axis=1)

x.boxplot()
plt.show()

scaler = MinMaxScaler()
x1 = scaler.fit_transform(x)
plt.boxplot(x1)
plt.show()

scaler2 = StandardScaler()
x2 = scaler2.fit_transform(x)
plt.boxplot(x2)
plt.show()

scaler3 = Binarizer()
x3 = scaler3.transform(x)
plt.boxplot(x3)
plt.show()