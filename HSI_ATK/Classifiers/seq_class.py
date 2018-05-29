import numpy as np
from scipy.stats import uniform, randint

from sklearn.decomposition import RandomizedPCA, PCA, FactorAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from mlens.ensemble import SequentialEnsemble
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from mlens.preprocessing import Subset


# from HSI_ATK.Generators.simple_gen import add_noise_2d
from HSI_ATK.Generators.gen3d import silly_gen


seed = 2018
np.random.seed(seed)

# image_set = np.genfromtxt('../TestData/c1_gn.csv', delimiter=',')
# label_set = np.genfromtxt('../TestData/c1_lb.csv', delimiter=',')
# l_space = np.genfromtxt('../TestData/c1_xy.csv', delimiter=',')
#
# image_set = add_noise_2d(image_set)
#
# x1train, x1test, y1train, y1test = train_test_split(image_set, label_set, test_size=0.12)
# x2train, x2test, y2train, y2test = train_test_split(image_set, l_space, test_size=0.12)

data_pix, spacial_pix = silly_gen(denoise=True)
X_train, X_test, y_train, y_test = train_test_split(data_pix, spacial_pix, test_size=.23, random_state=seed)

est_l1 = [
    #('rdc', OneVsRestClassifier(RidgeClassifier(tol=1e-4))),
    ('etr', OneVsRestClassifier(ExtraTreesClassifier(n_jobs=1))),
    ('rfr', OneVsRestClassifier(RandomForestClassifier(n_jobs=1))),
    ('mlp', OneVsRestClassifier(MLPClassifier())),
    ('svc', OneVsRestClassifier(SVC(tol=1e-4, degree=9))),
]

est_l2 = [
    ('rdc', OneVsRestClassifier(RidgeClassifier(tol=1e-4))),
    ('gbc', OneVsRestClassifier(GradientBoostingClassifier())),
    ('ada', OneVsRestClassifier(AdaBoostClassifier())),
    ('svc', OneVsRestClassifier(SVC(tol=1e-4, degree=9))),
    ('bag', OneVsRestClassifier(BaggingClassifier(n_jobs=1)))
]

ests_1 = {
    'case-1': est_l1,
    # 'case-2': est_l1,
    # 'case-3': est_l1,
    # 'case-4': est_l1
}

ests_2 = {
    'case-1': est_l2,
    # 'case-2': est_l2,
    # 'case-3': est_l2,
    # 'case-4': est_l2
}

r = uniform(0, 30)
d = randint(2, 10)
f = randint(100, 200)
e = uniform(0, 3)
ee = uniform(0, 1)

pars_1 = {
    'case-1.gbc': {},
    'case-1.mlp': {'alpha': ee,
                   'beta_1': ee,
                   'beta_2': ee,
                   'epsilon': ee},
    'case-1.ard': {'alpha_1': ee,
                   'alpha_2': ee,
                   'lambda_1': ee,
                   'lambda_2': ee},
    'case-1.rfr': {},
    'case-1.etr': {},
    'case-1.svc': {'C': e,
                   'degree': d,
                   'coef0': r,},
    'case-1.ada': {},
    'case-1.bag': {},
}

sc = StandardScaler()
pca = PCA(whiten=True)
fa = FactorAnalysis()

pre_cases = {
    'case-1': [],
    # 'case-2': [sc],
    # 'case-3': [pca],
    # 'case-4': [fa]
}

scorer = make_scorer(score_func=mean_absolute_error, greater_is_better=False, needs_proba=False, needs_threshold=False)

ensemble = SequentialEnsemble(model_selection=False, n_jobs=3, shuffle=True, random_state=seed, scorer=mean_squared_error)

ensemble.add('stack', ests_1, preprocessing=pre_cases)
ensemble.add('stack', ests_2, preprocessing=pre_cases)
ensemble.add_meta(OneVsRestClassifier(SVC(degree=9)))
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# ests = ensemble
# evaluator = Evaluator(scorer=scorer, random_state=seed, verbose=3, cv=2, n_jobs=1)
# evaluator.fit(X=data_pix, y=spacial_pix, estimators=ests, param_dicts=pars_1,
#               n_iter=5, preprocessing=pre_cases)
#
# print(evaluator.results)
