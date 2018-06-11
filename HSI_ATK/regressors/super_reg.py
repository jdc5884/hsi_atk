import numpy as np

# from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, accuracy_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

from mlens.metrics import make_scorer
from mlens.ensemble import SuperLearner


seed = np.random.seed(2018)

scorer = make_scorer(r2_score, greater_is_better=False)

def build_ensemble(incl_meta, meta_type='log', preprocessors=None, estimators=None, propagate_features=None):
    if propagate_features:
        n = len(propagate_features)
        propagate_features_1 = propagate_features
        propagate_features_2 = [i for i in range(n)]
    else:
        propagate_features_1 = propagate_features_2 = None

    if not estimators:
        estimators = [('rfr', RandomForestRegressor(random_state=seed)),
                      ('svr', SVR()),
                      ('rdg', Ridge())]

    ensemble = SuperLearner()
    ensemble.add(estimators, propagate_features=propagate_features_1)
    ensemble.add(estimators, propagate_features=propagate_features_2)

    if incl_meta & meta_type == 'log':
        ensemble.add_meta(LogisticRegression())
    elif incl_meta & meta_type == 'lin':
        ensemble.add_meta(LinearRegression())

    return ensemble

base = build_ensemble(True, 'log', [0, 1])