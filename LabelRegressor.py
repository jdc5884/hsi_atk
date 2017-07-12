__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import theil_sen
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import svm


hyper_data = pd.read_csv('massaged_data.csv', sep=',')

#hyper_data = pd.read_csv('massaged_data_nb.csv', sep=',')

# hyper_data.values[:, 35:37]+hyper_data.values[:, 51:57]+hyper_data.values[:, 59:71]+\
#  hyper_data.values[:, 99:101]+hyper_data.values[:, 127:129]

X = hyper_data.values[:, 13:]
y = hyper_data.values[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

clf = DecisionTreeRegressor(random_state=100, )
clf.fit(X_train, y_train)

#clf_ts = theil_sen.TheilSenRegressor()
#clf_ts.fit(X_train, y_train)

y_pred = clf.predict(X_test)
#y_pred_ts = clf_ts.predict(X_test)

print(accuracy_score(y_test, y_pred)*100)
#print(accuracy_score(y_test, y_pred_ts)*100)