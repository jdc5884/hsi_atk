import rasterio
import numpy as np

import sklearn.decomposition as dec
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression, RANSACRegressor, TheilSenRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from mlens.ensemble import SuperLearner


with rasterio.open("../Data/32.control.bil") as src:
    hsi_raw = np.array(src.read())

seed = np.random.seed(2017)


def build_ensemble(incl_meta, propagate_features=None):
    if propagate_features:
        n = len(propagate_features)
        propagate_features_1 = propagate_features
        propagate_features_2 = [i for i in range(n)]
    else:
        propagate_features_1 = propagate_features_2 = None

    estimators = [RandomForestRegressor(random_state=seed), SVR()]

    ensemble = SuperLearner()
    ensemble.add(estimators, propagate_features=propagate_features_1)
    ensemble.add(estimators, propagate_features=propagate_features_2)

    if incl_meta:
        ensemble.add_meta(LogisticRegression())

    return ensemble


base = build_ensemble(False)
# base.fit()