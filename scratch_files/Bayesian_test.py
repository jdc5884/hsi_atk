# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy import linalg, ndimage

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

hyper_data = pd.read_csv("../Data/headers3mgperml.csv", sep=',')

X = hyper_data.values[:, 15:]
y = hyper_data.values[:, 2]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=75)

cv = KFold(2)
ridge = BayesianRidge()
cachedir = tempfile.mkdtemp()
mem = Memory(cachedir=cachedir, verbose=1)
#
connectivity = grid_to_graph(n_x=240, n_y=34)
ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity, memory=mem)
#
clf = Pipeline([('ward', ward), ('ridge', ridge)])
clf = GridSearchCV(clf, {'ward__n_clusters': [10, 20, 30]}, n_jobs=1, cv=cv)
clf.fit(X,y)
# coef_ = clf.best_estimator_.steps[-1][1].coef_
# coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)
# coef_agglomeration_ = coef_.reshape(240, 34)

