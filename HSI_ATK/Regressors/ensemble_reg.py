import numpy as np
# import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from mlens.metrics import make_scorer
from mlens.ensemble import SuperLearner, SequentialEnsemble
from mlens.preprocessing import Subset
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from HSI_ATK.Generators.simple_gen import xy_gen, add_noise


image_set = []
label_set = []

n_sets = 25

for i in range(n_sets):
    im, la = xy_gen((50, 50), 4, 10, 5)
    image_set.append(im)
    label_set.append(la)

for j in range(n_sets):
    im, la = xy_gen((50, 50), 0, 2, 2)
    image_set.append(im)
    label_set.append(la)

for k in range(n_sets):
    im, la = xy_gen((50, 50), 4, 50, 5)
    image_set.append(im)
    label_set.append(la)

image_set = np.array(image_set)
image_set = add_noise(image_set)
imS = image_set.shape
image_set = image_set.reshape(imS[0], imS[1]*imS[2])

X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.33)

ests1 = [
    # ('las', Lasso(copy_X=True, max_iter=2000, normalize=True, positive=False)),
    ('rdg', Ridge(copy_X=True, max_iter=2000, normalize=False)),
    ('rfr', RandomForestRegressor()),
]

# ests2 = [
#     ('rfr', RandomForestRegressor()),
#     ('gbr', GradientBoostingRegressor())
# ]

n_iter = 10

preprocess_cases = {
    'none': [],
    'sc': [StandardScaler()],
    'sub': [Subset([0, 1])],
    # 'pca': [PCA()]
}

rpars = {
    'alpha': np.linspace(-2, 2, 10),
    'normalize': [True, False],
}

params = {
    # 'las': pars,
    'rdg': rpars,
}

ensemble = SuperLearner(scorer=r2_score)
ensemble.add(estimators=ests1)
ensemble.add_meta(LinearRegression())

ensemble.fit(X_train, y_train)

# ensemble.add(ests2)

ensemble.fit(X=X_train, y=y_train)

y_pred = np.array(ensemble.predict(X_test))
y_test = np.array(y_test)

score = r2_score(y_test, y_pred)
print(score)
