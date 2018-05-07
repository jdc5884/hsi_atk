import numpy as np
from scipy.stats import alpha

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlens.ensemble import SequentialEnsemble
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from mlens.preprocessing import Subset


from HSI_ATK.Generators.simple_gen import xy_gen, add_noise


a = 3.57
seed = 2018
np.random.seed(seed)

image_set = []
label_set = []

for i in range(25):
    im, la = xy_gen((50, 50), 4, 10, 5)
    image_set.append(im)
    label_set.append(la)

    im, la = xy_gen((50, 50), 0, 2, 2)
    image_set.append(im)
    label_set.append(la)

    im, la = xy_gen((50, 50), 4, 50, 5)
    image_set.append(im)
    label_set.append(la)

image_set = np.array(image_set)
image_set = add_noise(image_set)
imS = image_set.shape
image_set = image_set.reshape(imS[0], imS[1]*imS[2])

# X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.33)

preprocess_cases = {
    'none': [],
    'sc': [StandardScaler()],
    'sub': [Subset([0, 1])],
    'pca': [PCA()]
}

ests_1 = [
    ('rdg', Ridge()),
    ('rfr', RandomForestRegressor()),
]

ests_2 = [
    ('las', Lasso()),
    ('etr', ExtraTreesRegressor())
]

ests_3 = [
    ('svr', SVR()),
    # 'rdg': Ridge(),
]

pars_1 = {
    'alpha': alpha(a, size=20),
    'normalize': [True, False],
    'tol': np.logspace(-5, 3),
}

pars_2 = {
    'n_estimators': np.linspace(1, 15, 15),

}

params = {
    'none.rdg': pars_1,
    'sc.rdg': pars_1,
    'none.rfr': pars_2,
    'pca.rfr': pars_2,
    'sc.rfr': pars_2,
}

scorer = make_scorer(r2_score)

evaluator = Evaluator(scorer=scorer, cv=10, random_state=seed, verbose=3)
evaluator.fit(image_set, label_set, ests_1, params, 4, preprocess_cases)
print("\nComparison with different parameter dists:\n\n%r" % evaluator.results)

# ensemble = SequentialEnsemble()
# ensemble.add('blend', ests_1, params)
#
# params = {
#     'none.las': pars_1,
#     'sc.las': pars_1,
# }
#
# ensemble.add('blend', ests_2, params)
# ensemble.add('subsemble', ests_3, params)
# ensemble.add_meta(LinearRegression())
#
# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
