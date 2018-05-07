import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlens.model_selection import Evaluator
from mlens.metrics import make_scorer
from mlens.preprocessing import Subset


from HSI_ATK.Generators.simple_gen import xy_gen, add_noise


image_set = []
label_set = []

for i in range(50):
    im, la = xy_gen((50, 50), 4, 10, 5)
    image_set.append(im)
    label_set.append(la)

for j in range(50):
    im, la = xy_gen((50, 50), 0, 2, 2)
    image_set.append(im)
    label_set.append(la)

for k in range(50):
    im, la = xy_gen((50, 50), 4, 50, 5)
    image_set.append(im)
    label_set.append(la)

image_set = np.array(image_set)
image_set = add_noise(image_set)
imS = image_set.shape
image_set = image_set.reshape(imS[0], imS[1]*imS[2])

X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.33)

score_f = make_scorer(score_func=r2_score, greater_is_better=False)

evaluator = Evaluator(scorer=score_f, shuffle=False, verbose=True)

ests1 = [
    ('las', Lasso(copy_X=True, max_iter=2000, normalize=True, positive=False)),
    ('rdg', Ridge(copy_X=True, max_iter=2000, normalize=False))]
# ests2 = [('rfr', RandomForestRegressor()), ('gbr', GradientBoostingRegressor())]

n_iter = 10

preprocess_cases = {
    'none': [],
    'sc': [StandardScaler()],
    'sub': [Subset([0, 1])]
}

pars = {'alpha': np.logspace(-2, 2, n_iter)}
params = {'las':pars,
          'rdg':pars}

evaluator.fit(X_train, y_train, estimators=ests1, param_dicts=params, preprocessing=preprocess_cases, n_iter=n_iter)
print(evaluator.results())