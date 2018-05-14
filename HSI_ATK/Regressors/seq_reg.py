import numpy as np
from scipy.stats import uniform, randint

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlens.ensemble import SequentialEnsemble
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from mlens.preprocessing import Subset


from HSI_ATK.Generators.simple_gen import add_noise_2d


seed = 2018
np.random.seed(seed)

image_set = np.genfromtxt('../TestData/c1_gn.csv', delimiter=',')
label_set = np.genfromtxt('../TestData/c1_L_gn.csv', delimiter=',')

image_set = add_noise_2d(image_set)

X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.12)

ests_1 = [
    ('rdg', Ridge(max_iter=5000, random_state=seed)),
    ('las', Lasso(max_iter=5000, random_state=seed)),
]

ests_2 = [
#     ('rfr', RandomForestRegressor(random_state=seed)),
#     ('etr', ExtraTreesRegressor(random_state=seed)),
# ]
    ('ada', AdaBoostRegressor(random_state=seed)),
    ('bag', BaggingRegressor(n_jobs=2, random_state=seed))
]

ests_3 = [
    ('svr', SVR(kernel='linear', max_iter=5000)),
]

r = uniform(0, 10)
d = randint(2, 10)
f = randint(1,100)
e = uniform(0, 1)

pars_1 = {
    'ens': {
        'rdg': {'alpha': r,
                # 'normalize': [True, False],
                # 'solver': ['auto', 'svd', 'cholesky',
                #            'lsqr', 'sparse_cg', 'sag', 'saga'],
                },
        'las': {'alpha': r,
                # 'normalize': [True, False],
                # 'selection': ['cyclic', 'random'],
                # 'warm_start': [True, False],
                # 'positive': [True, False]
                },
        'rfr': {'n_estimators': f,
                'max_features': f,
                },
        'etr': {'n_estimators': f,
                'max_features': f,
                },
        'svr': {'C': r,
                'epsilon': e,
                # 'kernel': ['linear', 'poly', 'rbf',
                #            'sigmoid', 'precomputed'],
                'degree': d,
                'coef0': r,
                # 'shrinking': [True, False]
                },
        'ada': {'n_estimators': f,
                'learning_rate': e},
        'bag': {'n_estimators': f,
                }
    }
}

sc = StandardScaler()
preprocessing = [
    ('sc', [sc]),
]

scorer = make_scorer(r2_score, greater_is_better=False, needs_proba=False)

# evaluator = Evaluator(scorer=scorer, cv=2, random_state=seed, verbose=3)
# evaluator.fit(image_set, label_set, ests_1, n_iter=4)
# print("\nComparison with different parameter dists:\n\n%r" % evaluator.results)

ensemble = SequentialEnsemble(model_selection=True, n_jobs=1)
ensemble.add('blend', ests_1, preprocessing=sc)
ensemble.add('blend', ests_2)
ensemble.add('stack', ests_3, preprocessing=sc)
ensemble.add_meta(Ridge(max_iter=5000))

# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
# print(r2_score(y_test, y_pred))
#
evaluator = Evaluator(scorer=scorer, random_state=seed, verbose=True, n_jobs=1)
# ests = [ensemble]
evaluator.fit(image_set, label_set, [('ens', ensemble)], pars_1, 40, [sc])
print(evaluator.results)
print(evaluator.metrics)
