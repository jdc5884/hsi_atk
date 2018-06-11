import numpy as np
from scipy.stats import uniform

# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, Ridge, Lasso
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator


seed = 2018
np.random.seed(seed)

# image_set = np.genfromtxt('../testdata/c1_gn.csv', delimiter=',')
# label_set = np.genfromtxt('../testdata/c1_L_gn.csv', delimiter=',')

r2_scorer = make_scorer(r2_score)

ests = [('rdg', Ridge(max_iter=4000)), ('las', Lasso(max_iter=4000))]

a = uniform(0, 10)

params = {
    'rdg': {'alpha': a},
    'las': {'alpha': a}
}

preproc = {
    'none': [],
    'sc': [StandardScaler()]
}

evaluator = Evaluator(r2_scorer, cv=2, random_state=seed, verbose=1)
# evaluator.fit(image_set, label_set, ests, params, 40, preproc)
# print(evaluator.results)

def sig(z):
    return 1/(1 + np.exp(-z))

def hyp(th, x):
    return sig(x @ th)

def cost_func(x, y, th, m):
    hi = hyp(th, x)
    y_ = y.reshape(-1, 1)
    j = 1/float(m) * np.sum(-y_ * np.log(hi) - (1 - y_) * np.log(1 - hi))
    return j

def cost_func_dx(x, y, th, m, alpha):
    hi = hyp(th, x)
    y_ = y.reshape(-1, 1)
    j = alpha/float(m) * x.T @ (hi - y_)
    return j

def grad_desc(x, y, th, m, alpha):
    n_th = th - cost_func_dx(x, y, th, m, alpha)
    return n_th

# def acc(th):
#     correct = 0
#     length = len(X_test)
#     pred = (hyp(th, X_test) > 0.5)
#     y_ = y_test.reshape(-1, 1)
#     # create threshold score
#     correct = pred == y_
#     accuracy = (np.sum(correct) / length) * 100
#     return accuracy



# evaluator.fit(image_set, label_set, ests, n_iter=10)
#
# df = evaluator.results