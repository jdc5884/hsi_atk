# author David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import random
import numpy as np
import pandas as pd

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
import xgboost as xgb

hyper_data = pd.read_csv('../Data/headers3mgperml.csv', sep=',')

X = hyper_data.values[:, 15:]
y = hyper_data.values[: 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.25)

paramgrid = {
    'kernel': ['rbf'],
    'C': np.logspace(-9, 9, num=25, base=10),
    'gamma': np.logspace(-9, 9, num=25, base=10)
}

random.seed(1)

cv = EvolutionaryAlgorithmSearchCV(estimator=RandomForestClassifier(),
                                   params=paramgrid,
                                   scoring='accuracy',
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)

cv.fit(X_train, y_train)
