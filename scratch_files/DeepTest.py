# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from sklearn import model_selection

hyper_data = pd.read_csv("../Data/headers3mgperml.csv", sep=',')

X = hyper_data.values[:, 15:]
y = hyper_data.values[:, 2]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=90, test_size=0.25)

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))