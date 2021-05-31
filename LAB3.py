# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:02:37 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
# iris.feature_names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['species'] = iris.target

#x and y split
y = df.species
x = df.drop('species', axis=1)

#train and test split
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=1)

# Build DT, RF, LDA, kNN, and SVM 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

models = [DecisionTreeClassifier(),RandomForestClassifier(),LinearDiscriminantAnalysis(),KNeighborsClassifier(), SVC()]

scores = []
cvs = []
for model in models:
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    accuracy_score(y_test, y_pred)
    cvs.append(cross_val_score(model, x_test, y_test, scoring='accuracy', cv=10))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

df_cvs = pd.DataFrame(cvs).T
df_cvs.columns = ['DT', 'RF', 'LDA', 'kNN', 'SVM']
df_cvs.boxplot()

summary_df = pd.concat([df_cvs.mean(), df_cvs.std()], axis=1)
summary_df.columns = ['mean', 'std']
print(summary_df)

#prediction
x_train
test_data = pd.DataFrame({'sepal_length':6.2,
                          'sepal_width':3.1,
                          'petal_length':4.3,
                          'petal_width':1.1}, index=[0])
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
lda.predict(test_data)
