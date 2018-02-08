import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets as ds
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

clf = lm.Ridge()

n_samples = 1000
n_outliers = 50

X, y, w = ds.make_regression(n_samples=1000, n_features=240, n_informative=10,
                             n_targets=1, noise=30, coef=True, bias=3.5,
                             random_state=1)

coefs = []
errors = []

alphas = np.logspace(-6,6,200)

for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X,y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, w))

plt.figure(figsize=(20,6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.show()

# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)
#
# lr = lm.LinearRegression()
# lr.fit(X,y)
#
# ransac = lm.RANSACRegressor()
# ransac.fit(X,y)
# inlier_m = ransac.inlier_mask_
# outlier_m = np.logical_not(inlier_m)
#
# line_X = np.arange(X.min(), X.max())[:, np.newaxis]
#
# print(X, '\n\n', y, '\n\n', line_X)
#
# line_y = lr.predict(line_X)
# line_y_ransac = ransac.predict(line_X)

