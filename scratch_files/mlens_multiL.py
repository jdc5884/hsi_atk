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

# Building an ensemble
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Multi-layer ensembles ---

ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)

# Build first layer
ensemble.add([RandomForestClassifier(random_state=seed), LogisticRegression()])

# Build the second layer
ensemble.add([LogisticRegression(), SVC()])

# Attach final meta estimator
ensemble.add_meta(SVC())

ensemble.fit(X[:75], y[:75])
preds = ensemble.predict(X[75:])
print("Fit data:\n%r" % ensemble.data)