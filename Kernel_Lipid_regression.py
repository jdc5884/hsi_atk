# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

hyper_data = pd.read_csv('headers3mgperml.csv', sep=',')

X = hyper_data.values[:, 15:]
y1 = hyper_data.values[:, 5]
y2 = hyper_data.values[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=100, test_size=0.3)

clf = svm.SVR()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, random_state=100, test_size=0.3)

clf2 = svm.SVR()
clf2.fit(X_train2, y_train2)

y_pred2 = clf2.predict(X_test)
print(accuracy_score(y_test2, y_pred2))