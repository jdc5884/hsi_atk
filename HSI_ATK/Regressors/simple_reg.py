import numpy as np
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlens.metrics import make_scorer
# from mlens.model_selection import Evaluator
from pandas import DataFrame

from HSI_ATK.Generators.simple_gen import *

f1_scorer = make_scorer(f1_score, average='micro', greater_is_better=True)

# ests = [("lrg", LinearRegression()), ("trg", TheilSenRegressor()), ("rrg", RANSACRegressor())]

# evaluator = Evaluator(f1_scorer, cv=10, random_state=32, verbose=1)

image_set = []
label_set = []

for i in range(500):
    im, la = xy_gen((100,100), 20)
    image_set.append(np.array(im))
    label_set.append(la)

image_set = np.array(image_set)
imS = image_set.shape
print(imS)
image_set = image_set.reshape(imS[0], imS[1]*imS[2])


## volume test
X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=0.25)

# reg = LinearRegression(normalize=True)
reg = RANSACRegressor(min_samples=80)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
acc = f1_scorer(y_pred=y_pred, y_true=y_test)

# print(acc)

# evaluator.fit(image_set, label_set, ests, n_iter=10)
#
# df = evaluator.results