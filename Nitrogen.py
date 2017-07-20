__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import svm

#from HyperspecClassifier import *

# Read in data to pandas csv object
#hyper_data = pd.read_csv('headers3mgperml.csv', sep=',')
hyper_data = pd.read_csv('massaged_data.csv', sep=',')

# Specifying data index slices
X = hyper_data.values[:, 13:]     # Signal data from specific wavelengths
Y = hyper_data.values[:, 3]             # Label data from Density values

# Separating training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=75)

# X_train[:, 121:123][1:] *= 5
# X_train[:, 144][1:] *= 5
# X_train[:, 185][1:] *= 15

# clf = svm.SVC()
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)


# attempting to add weighting
f_score, p_val = f_regression(X_train, y_train)


clf = DecisionTreeClassifier()

#clf = HyperspecClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred)
print(accuracy_score(y_test, y_pred))

#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
#
# def plot_decision_function(classifier, sample_weight, axis, title):
#     # plot the decision function
#     xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
#
#     Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     # plot the line, the points, and the nearest vectors to the plane
#     axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
#     axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
#                  cmap=plt.cm.bone)
#
#     axis.axis('off')
#     axis.set_title(title)
#
#
# # we create 20 points
# np.random.seed(0)
# #X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# #y = [1] * 10 + [-1] * 10
# y = Y
# sample_weight_last_ten = abs(np.random.randn(len(X)))
# sample_weight_constant = np.ones(len(X))
# # and bigger weights to some outliers
# sample_weight_last_ten[15:] *= 5
# sample_weight_last_ten[9] *= 15
#
# # for reference, first fit without class weights
#
# # fit the model
# clf_weights = svm.SVC()
# clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)
#
# clf_no_weights = svm.SVC()
# clf_no_weights.fit(X, y)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
#                        "Constant weights")
# plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
#                        "Modified weights")
#
# plt.show()
