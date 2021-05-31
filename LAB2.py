# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:34:02 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import dataests

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
iris.feature_names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['class'] = iris.target
df.head()

# X and y split. 독립, 종속변수 분리
y = df['class']
x = df.drop('class', axis=1)

# Splitting Dataset (test size=.2). 데이터셋 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

# Decision Tree 의사결정트리
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train, y_train)

# Model evaluation (accuracy score) 모델평가
dt.score(x_test, y_test)
from sklearn.metrics import accuracy_score
y_pred = dt.predict(x_test)
dt_score = accuracy_score(y_test, y_pred)

# 5 fold cross validation (5 fold) 교차검증
from sklearn.model_selection import cross_val_score
dt_cv = cross_val_score(dt, x_test, y_test, cv=5, scoring='accuracy')
dt_cv.mean(), dt_cv.std()

# Decision tree visualization (max depth=3) 의사결정트리 시각화
from IPython.display import Image  
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(dt, out_file=None,feature_names=iris.feature_names,                                class_names=iris.target_names, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# Random Forest 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Model evaluation (accuracy score) 모델평가
rf.score(x_test, y_test)
y_pred = rf.predict(x_test)
rf_score = accuracy_score(y_test, y_pred)

# 5 fold cross validation (5 fold) 5배 교차검증
rf_cv = cross_val_score(rf, x_test, y_test, cv=5, scoring='accuracy')
rf_cv
rf_cv.mean(), rf_cv.std()

#comparison
means = [dt_cv.mean(), rf_cv.mean()]
stds = [dt_cv.std(), rf_cv.std()]
df1 = pd.DataFrame({'mean': means,
              'std': stds}, index=['DT', 'RF'])
print(df1)

df2 = pd.DataFrame({'DT': dt_cv,
                    'RF': rf_cv})
df2
df2.boxplot()
plt.show()
