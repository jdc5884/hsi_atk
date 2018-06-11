import numpy as np
from scipy.stats import uniform, randint

from mlens.ensemble import SequentialEnsemble
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR, SVC

from HSI_ATK.Generators.gen3d import silly_gen


def build_ensemble(**kwargs):
    sc = StandardScaler()
    pca = PCA()
    # pf = PolynomialFeatures(degree=5)
    ests_1 = [
        ('rfr', RandomForestRegressor(n_estimators=5)),
        ('rdg', Ridge(tol=1e-4, max_iter=4000)),
        ('mlr', MLPRegressor((100, 20), max_iter=1000))
        ]
    ests_2 = [
        ('rdg', Ridge(tol=1e-4, max_iter=4000)),
        ('svr', SVR(tol=1e-4, kernel='linear', degree=5, max_iter=4000)),
        # ('etc', ExtraTreesClassifier(n_estimators=15))
    ]

    ensemble = SequentialEnsemble(**kwargs, shuffle=False)
    ensemble.add("blend", ests_1, preprocessing=[sc], )
    # ensemble.add("subsemble", [LinearRegression()])
    ensemble.add("stack", ests_2, preprocessing=[sc])
    ensemble.add_meta([('etc', ExtraTreesClassifier(n_estimators=5))])

    return ensemble


data_pix, spacial_pix, data, spacial_data = silly_gen(denoise=False)
indices = np.random.permutation(data_pix.shape[0])
training_idx, test_idx = indices[:2200], indices[2200:]
X_train, X_test = data_pix[training_idx, :], data_pix[test_idx, :]
y_train, y_test = spacial_pix[training_idx], spacial_pix[test_idx]

a = uniform(0, 4)
f = randint(2, 30)
C = uniform(0, 2)
e = uniform(0, 1)
d = randint(3, 9)

pars = {}

pre_cases = {
    'ens.sequentialensemble': [StandardScaler()]
}

ensemble = build_ensemble()
est = ensemble
# scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# evaluator = Evaluator(scorer=scorer, shuffle=False, verbose=True)
# evaluator.fit(data_pix, spacial_pix, est, pars, n_iter=4, preprocessing=pre_cases)
# print(evaluator.results)
ensemble.fit(X_train, y_train)
preds = ensemble.predict(X_test)

print(confusion_matrix(y_test, preds))
