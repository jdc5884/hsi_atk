# import numpy as np
from scipy.stats import uniform, randint

# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

from mlens.model_selection import Evaluator
from mlens.metrics import make_scorer
# from mlens.preprocessing import Subset


# image_set = np.genfromtxt('../testdata/c1_gn.csv', delimiter=',')
# label_set = np.genfromtxt('../testdata/c1_L_gn.csv', delimiter=',')
#
# X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.33)

score_f = make_scorer(score_func=r2_score, greater_is_better=False)

evaluator = Evaluator(scorer=score_f, shuffle=True, verbose=True)

estimators = [
    ('las', Lasso(copy_X=True, max_iter=4000)),
    ('rdg',Ridge(copy_X=True, max_iter=4000)),
    # ('rfr', RandomForestRegressor()),

]

params = {
    'las': {'alpha': uniform(0,5)},
    'rdg': {'alpha': uniform(0,5)},
    'rfr': {'n_estimators': randint(2,10), 'max_depth': randint(2,10),
            'max_features': uniform(0.5, 0.5)}
}

# n_iter = 20
# evaluator.fit(image_set, label_set, estimators=estimators, param_dicts=params, n_iter=n_iter)
# print(evaluator.results, '\n\n', evaluator.params)
