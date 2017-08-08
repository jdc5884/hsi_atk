# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score


#hyper_data = pd.read_csv("headers3mgperml.csv", sep=',')
hyper_data = pd.read_csv("../Data/massaged_data.csv", sep=',')

X = hyper_data.values[:, 13:]
y = hyper_data.values[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# MLP Neural Net. solver to 'lbfgs' for small datasets
# use solver: 'adam' for large datasets
clf = MLPRegressor(activation='logistic', solver='lbfgs',)

f_score, p_val = f_regression(X_train, y_train)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_test, '\n', y_pred)

print(f_score, '\n', p_val)