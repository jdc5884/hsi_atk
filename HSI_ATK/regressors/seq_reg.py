import numpy as np
from scipy.stats import uniform, randint

# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import RandomizedPCA, PCA
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression, \
    BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, ARDRegression, MultiTaskElasticNet, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
    # explained_variance_score, f1_score, average_precision_score, accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

from mlens.ensemble import SequentialEnsemble, SuperLearner
# from mlens.metrics import make_scorer
# from mlens.model_selection import Evaluator
# from mlens.preprocessing import Subset


# from HSI_ATK.generators.simple_gen import add_noise_2d
# from HSI_ATK.generators.gen3d import silly_gen


seed = np.random.seed(2018)

# image_set = np.genfromtxt('../testdata/c1_gn.csv', delimiter=',')
# label_set = np.genfromtxt('../testdata/c1_lb.csv', delimiter=',')
# l_space = np.genfromtxt('../testdata/c1_xy.csv', delimiter=',')

# image_set = add_noise_2d(image_set)

# x1train, x1test, y1train, y1test = train_test_split(image_set, label_set, test_size=0.12)
# x2train, x2test, y2train, y2test = train_test_split(image_set, l_space, test_size=0.12)

# data_pix, spacial_pix, data, spacial_data = silly_gen()
# X_train, X_test, y_train, y_test = train_test_split(data_pix, spacial_pix, test_size=.23, random_state=seed)

ests = {
    'case-1': [#('lin', LinearRegression()),
               ('bay', BayesianRidge(tol=1e-5)),
               #('hub', HuberRegressor(max_iter=5000, tol=1e-5)),
               ('ard', ARDRegression(tol=1e-5)),
               ('par', PassiveAggressiveRegressor(max_iter=5000, tol=1e-5)),
               ('rdg', Ridge(max_iter=5000, random_state=seed)),
               ('las', Lasso(max_iter=5000, random_state=seed)),
               # ('eln', ElasticNet(max_iter=5000, tol=1e-5, random_state=seed)),
               ('svr', SVR(kernel='linear')),
               ('mlp', MLPRegressor())]
}

r = uniform(0, 30)
d = randint(2, 10)
f = randint(1,100)
e = uniform(0, 3)
ee = uniform(0, 1)

pars_1 = {
    'case-1.mlp': {'alpha': ee,
                   'beta_1': e,
                   'beta_2': e,
                   'epsilon': ee},
    'case-1.eln': {'alpha': e,
                   'l1_ratio': e},
    'case-1.par': {'C': e,
                   'epsilon': e},
    'case-1.ard': {'alpha_1': ee,
                   'alpha_2': ee,
                   'lambda_1': ee,
                   'lambda_2': ee},
    'case-1.hub': {'epsilon': e,
                   'alpha': e},
    'case-1.bay': {'alpha_1': ee,
                   'alpha_2': ee,
                   'lambda_1': ee,
                   'lambda_2': ee},
    'case-1.rdg': {'alpha': e,},
    'case-1.las': {'alpha': e,},
    'case-1.rfr': {'n_estimators': f,
                   'max_features': f,},
    'case-1.etr': {'n_estimators': f,
                   'max_features': f,},
    'case-1.svr': {'C': e,
                   'epsilon': e,
                   'degree': d,
                   'coef0': r,},
    'case-1.ada': {'n_estimators': f,
                   'learning_rate': e},
    'case-1.bag': {'n_estimators': f,}

}

pre_cases = {
    # 'case-1': [],
    'case-1': [StandardScaler()]
}

# scorer = make_scorer(r2_score, greater_is_better=False, needs_proba=False, needs_threshold=False)

ensemble = SequentialEnsemble(model_selection=False, n_jobs=3, shuffle=True, random_state=seed, scorer=mean_absolute_error)
ensemble.add('blend', ests, preprocessing=pre_cases)
ensemble.add_meta(SVR())

# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
# print(mean_absolute_error(y_test, y_pred))

# ests = [ensemble]
# evaluator = Evaluator(scorer=scorer, random_state=seed, verbose=3, cv=4, n_jobs=1)
# evaluator.fit(X=image_set, y=label_set, estimators=ests, param_dicts=pars_1,
#               n_iter=40, preprocessing=pre_cases)
#
# print(evaluator.results)
