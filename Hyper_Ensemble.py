# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from LearnHelper import *

from evolutionary_search import maximize

train = pd.read_csv("~/PycharmProjects/HyperSpectralImaging/Data/massaged_data_train.csv")
test = pd.read_csv("~/PycharmProjects/HyperSpectralImaging/Data/massaged_data_test.csv")

GenoTypes = test['Genotype']

hyper_data = [train, test]

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(NFOLDS)


def col_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[train_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': False,
    'max_features': 0.05,
    'max_depth': 6,
    'min_samples_leaf': 8,
    # 'max_features': 'sqrt',
    'verbose': 0,
}

et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_features': 0.05,
    'max_depth': 8,
    'min_samples_leaf': 8,
    'verbose': 0,
}

ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.9,
}

gb_params = {
    'n_estimators': 500,
    'max_features': 0.05,
    'max_depth': 4,
    'min_samples_leaf': 8,
    'verbose': 0,
}

svc_params = {
    # 'kernel': 'linear',
    'C': 0.025,
}

rf = LearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = LearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = LearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = LearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = LearnHelper(clf=SVC, seed=SEED, params=svc_params)

X_train = train.values[:, 13:]
y_train = train.values[:, 1]
X_test = test.values[:, 13:]
y_test = test.values[:, 1]

et_oof_train, et_oof_test = col_oof(et, X_train, y_train, X_test)
rf_oof_train, rf_oof_test = col_oof(rf, X_train, y_train, X_test)
ada_oof_train, ada_oof_test = col_oof(ada, X_train, y_train, X_test)
gb_oof_train, gb_oof_test = col_oof(gb, X_train, y_train, X_test)
svc_oof_train, svc_oof_test = col_oof(svc, X_train, y_train, X_test)

print("clf's trained..")

rf_feature = rf.feature_important(X_train, y_train)
et_feature = et.feature_important(X_train, y_train)
ada_feature = ada.feature_important(X_train, y_train)
gb_feature = gb.feature_important(X_train, y_train)

base_predictions_train = pd.DataFrame({
    'RandomeForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel(),
})
base_predictions_train.head()

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    learning_rate=0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackSub = pd.DataFrame({
    'GenoTypes': GenoTypes,
    'GenoPreds': predictions,
})
StackSub.to_csv("~/PycharmProjects/HyperSpectralImaging/Output/StackSub.csv", index=False)

print(y_test)
print(predictions)


def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
best_params, best_score, score_results = maximize(func, param_grid, args, verbose=False)

print(best_params, best_score, score_results)
