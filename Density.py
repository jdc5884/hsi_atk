__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm


# Read in data to pandas csv object
hyper_data = pd.read_csv('headers3mgperml.csv', sep=',')

# Specifying data index slices
X = hyper_data.values[:, 15:]           # Signal data from specific wavelengths
Y = hyper_data.values[:, 2]             # Label data from Density values

# Separating training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Creating gini DecisionTree
clf_gini = DecisionTreeClassifier(random_state=100, max_depth=3,
                                  min_samples_leaf=5)
clf_gini.fit(X_train, y_train)          # Training gini tree

# Creating entropy DecisionTree
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)       # Training entropy tree

# Creating SVM with polynomial kernel
clf_svc = svm.SVC(random_state=100, kernel='poly')
clf_svc.fit(X_train, y_train)           # Training SVM


y_pred_gi = clf_gini.predict(X_test)    # gini tree prediction test
y_pred_en = clf_entropy.predict(X_test) # entropy tree prediction test
y_pred_sv = clf_svc.predict(X_test)     # SVM prediction test

# Print accuracy scores
print("Gini accuracy score: ", accuracy_score(y_test, y_pred_gi)*100)
print("Entropy accuracy score: ", accuracy_score(y_test, y_pred_en)*100)
print("SVM accuracy score: ", accuracy_score(y_test, y_pred_sv)*100)