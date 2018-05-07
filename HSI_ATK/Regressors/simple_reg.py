import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score
from mlens.metrics import make_scorer
# from mlens.model_selection import Evaluator
from pandas import DataFrame

from HSI_ATK.Generators.simple_gen import *

f1_scorer = make_scorer(f1_score, average='micro', greater_is_better=True)

# ests = [("lrg", LinearRegression()), ("trg", TheilSenRegressor()), ("rrg", RANSACRegressor())]

# evaluator = Evaluator(f1_scorer, cv=10, random_state=32, verbose=1)

image_set = []
label_set = []

for i in range(50):
    im, la = xy_gen((20, 20), 4, 5, 5)
    image_set.append(im)
    label_set.append(la)

for j in range(50):
    im, la = xy_gen((20, 20), 0, 2, 2)
    image_set.append(im)
    label_set.append(la)

for k in range(50):
    im, la = xy_gen((20, 20), 4, 50, 5)
    image_set.append(im)
    label_set.append(la)

image_set = np.array(image_set)
imS = image_set.shape
print(imS)
image_set = image_set.reshape(imS[0], imS[1]*imS[2])


## volume test
X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.23)

# reg = LinearRegression(normalize=True)
reg = Lasso(random_state=32)
# reg = RANSACRegressor(base_estimator=None, min_samples=10)
# reg = TheilSenRegressor()
# reg = BaggingRegressor()
# reg = RandomForestRegressor()
# reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

y_test = np.around(y_test, decimals=0)
y_pred = np.around(y_pred, decimals=0)

score = r2_score(y_test, y_pred)

print(score)

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

def acc(th):
    correct = 0
    length = len(X_test)
    pred = (hyp(th, X_test) > 0.5)
    y_ = y_test.reshape(-1, 1)
    # create threshold score
    correct = pred == y_
    accuracy = (np.sum(correct) / length) * 100
    return accuracy



# evaluator.fit(image_set, label_set, ests, n_iter=10)
#
# df = evaluator.results