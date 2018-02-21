import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

seed = 2017
np.random.seed(seed)

data = load_iris()
idx = np.random.permutation(150)
X = data.data[idx]
y = data.target[idx]

from mlens.metrics import make_scorer
accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)

from mlens.model_selection import Evaluator
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

# name estimators
ests = [('gnb', GaussianNB()), ('knn', KNeighborsClassifier())]

# map parameters to these, gnb has none
pars = {'n_neighbors': randint(2, 20)}
params = {'knn':pars}

# running evaluation over ests and param distributions by fit
evaluator = Evaluator(accuracy_scorer, cv=10, random_state=seed, verbose=1)
evaluator.fit(X, y, ests, params, n_iter=10)

print("Score comparison with best params found:\n\n%r" % evaluator.results)

# --- Pre-processing pipelines ---

from mlens.preprocessing import Subset
from sklearn.preprocessing import StandardScaler

# Map preprocessing cases through a dictionary
preprocess_cases = {
    'none': [],
    'sc': [StandardScaler()],
    'sub': [Subset([0, 1])]
}

evaluator.fit(X, y, preprocessing=preprocess_cases)

# --- Model selection across pre-processing pipelines
evaluator.fit(X, y, ests, params, n_iter=10)
print("\nComparison across preprocessing piplines:\n\n%r" % evaluator.results)

pars_1 = {'n_neighbors': randint(20, 30)}
pars_2 = {'n_neighbors': randint(2, 10)}
params = {'sc.knn':pars_1,
          'none.knn':pars_2,
          'sub.knn':pars_2}

# Map different estimators to different cases
ests_1 = [('gnb', GaussianNB()), ('knn', KNeighborsClassifier())]
ests_2 = [('knn', KNeighborsClassifier())]
estimators = {'sc': ests_1,
              'none': ests_2,
              'sub': ests_1}

evaluator.fit(X, y, estimators, params, n_iter=10)
print("\nComparison with different parameter dists:\n\n%r" % evaluator.results)

