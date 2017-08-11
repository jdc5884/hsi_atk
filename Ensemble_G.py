# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as plty
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from LearnHelper import *

from evolutionary_search import maximize

plty.offline.init_notebook_mode()

train = pd.read_csv("../Data/massaged_data_train.csv")
test = pd.read_csv("../Data/massaged_data_test.csv")

Density = test['Density']

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
y_train = train.values[:, 2]
X_test = test.values[:, 13:]

# with kfold
et_oof_train, et_oof_test = col_oof(et, X_train, y_train, X_test)
rf_oof_train, rf_oof_test = col_oof(rf, X_train, y_train, X_test)
ada_oof_train, ada_oof_test = col_oof(ada, X_train, y_train, X_test)
gb_oof_train, gb_oof_test = col_oof(gb, X_train, y_train, X_test)
svc_oof_train, svc_oof_test = col_oof(svc, X_train, y_train, X_test)

# with stratified kfold
# et_oof_train_s, et_oof_test_s = col_oof_s(et, X_train, y_train, X_test)
# rf_oof_train_s, rf_oof_test_s = col_oof_s(rf, X_train, y_train, X_test)
# ada_oof_train_s, ada_oof_test_s = col_oof_s(ada, X_train, y_train, X_test)
# gb_oof_train_s, gb_oof_test_s = col_oof_s(gb, X_train, y_train, X_test)
# svc_oof_train_s, svc_oof_test_s = col_oof_s(svc, X_train, y_train, X_test)

print("clf's trained..")

rf_feature = rf.feature_important(X_train, y_train)
et_feature = et.feature_important(X_train, y_train)
ada_feature = ada.feature_important(X_train, y_train)
gb_feature = gb.feature_important(X_train, y_train)

# TODO: put feature to python lists for weighting
# TODO: may be able to use tree attribute/method 'feature_importances_'
# TODO: which returns arrays with class weights

rf_features = [1.75178754e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.02773723e-04,
               1.00350097e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 1.49087750e-05, 2.00000000e-03, 0.00000000e+00,
               2.00000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               2.00000000e-03, 2.00000000e-03, 0.00000000e+00, 0.00000000e+00,
               2.00000000e-03, 1.97880073e-03, 2.00000000e-03, 2.78260870e-04,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 4.10958904e-05, 0.00000000e+00, 1.60494795e-03,
               2.00000000e-03, 9.83675362e-03, 2.00000000e-03, 0.00000000e+00,
               0.00000000e+00, 6.00000000e-03, 3.78936170e-03, 2.00000000e-03,
               4.00000000e-03, 2.00000000e-03, 8.02531245e-03, 1.60000000e-02,
               8.00000000e-03, 1.80000000e-02, 1.80000000e-02, 2.20000000e-02,
               1.62482125e-02, 1.60000000e-02, 1.20000000e-02, 1.40000000e-02,
               1.80000000e-02, 1.01855458e-02, 2.06558018e-02, 6.51898941e-03,
               1.40000000e-02, 1.53817076e-02, 8.21063830e-03, 4.00000000e-03,
               4.10627074e-03, 2.00000000e-03, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 3.26457573e-05, 0.00000000e+00, 2.00000000e-03,
               0.00000000e+00, 0.00000000e+00, 2.47341083e-05, 0.00000000e+00,
               2.00000000e-03, 0.00000000e+00, 3.86547701e-05, 0.00000000e+00,
               0.00000000e+00, 3.26457573e-05, 2.00000000e-03, 5.51132884e-04,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.00000000e-03,
               2.00000000e-03, 0.00000000e+00, 0.00000000e+00, 2.00000000e-03,
               2.00000000e-03, 0.00000000e+00, 2.09859155e-03, 2.00000000e-03,
               4.00000000e-03, 2.00000000e-03, 5.97945205e-03, 0.00000000e+00,
               4.00000000e-03, 4.00000000e-03, 4.00000000e-03, 6.00000000e-03,
               0.00000000e+00, 2.02054795e-03, 2.00000000e-03, 2.00000000e-03,
               2.00000000e-03, 4.16324638e-03, 2.00000000e-03, 0.00000000e+00,
               5.01334445e-03, 2.00000000e-03, 4.00000000e-03, 2.00000000e-03,
               4.00000000e-03, 2.02119927e-03, 0.00000000e+00, 0.00000000e+00,
               2.00000000e-03, 4.00000000e-03, 5.95890411e-03, 0.00000000e+00,
               1.77650597e-04, 2.00000000e-03, 4.00000000e-03, 4.00000000e-03,
               0.00000000e+00, 2.00000000e-03, 0.00000000e+00, 2.00000000e-03,
               2.00000000e-03, 2.02119927e-03, 4.00000000e-03, 2.00000000e-03,
               1.19643917e-04, 4.00000000e-03, 2.00000000e-03, 0.00000000e+00,
               6.00000000e-03, 2.00000000e-03, 4.00000000e-03, 2.00000000e-03,
               2.00000000e-03, 0.00000000e+00, 1.00000000e-02, 2.01805632e-03,
               2.00000000e-03, 6.00000000e-03, 0.00000000e+00, 1.00000000e-02,
               5.97526589e-03, 1.44886712e-03, 1.00000000e-02, 1.20000000e-02,
               5.98194368e-03, 4.00000000e-03, 2.00000000e-03, 4.00000000e-03,
               1.20000000e-02, 5.97880073e-03, 1.18617061e-02, 7.96735424e-03,
               1.00000000e-02, 2.00000000e-02, 2.18223494e-02, 1.20000000e-02,
               5.96134523e-03, 1.20000000e-02, 1.89722628e-03, 1.49000000e-02,
               7.98509123e-03, 6.00000000e-03, 8.00000000e-03, 5.96735424e-03,
               4.00000000e-03, 0.00000000e+00, 1.60000000e-02, 4.00000000e-03,
               6.00000000e-03, 6.00000000e-03, 2.00000000e-03, 6.00000000e-03,
               0.00000000e+00, 6.00000000e-03, 4.00000000e-03, 8.00000000e-03,
               5.89372926e-03, 0.00000000e+00, 4.00000000e-03, 1.12056518e-02,
               4.00000000e-03, 1.00000000e-02, 1.40000000e-02, 4.00000000e-03,
               4.61920688e-03, 1.40000000e-02, 8.00000000e-03, 7.89964990e-03,
               4.00000000e-03, 1.20000000e-02, 3.88035608e-03, 2.00000000e-03,
               2.00000000e-03, 2.00000000e-03, 2.00000000e-03, 8.00000000e-03,
               4.10035010e-03, 6.00000000e-03, 1.38144542e-02, 9.97468755e-03,
               9.72173913e-03, 8.00000000e-03, 0.00000000e+00, 6.00000000e-03,
               6.00000000e-03, 4.00000000e-03, 0.00000000e+00, 2.00000000e-03,
               6.00000000e-03, 4.00000000e-03, 4.00000000e-03, 2.00000000e-03,
               8.00000000e-03, 2.13829394e-03, 0.00000000e+00, 2.00000000e-03,
               4.00000000e-03, 8.00000000e-03, 0.00000000e+00, 1.00000000e-02,
               2.00000000e-03, 0.00000000e+00, 4.00000000e-03, 7.90140845e-03]
et_features = [4.51127820e-05, 4.24250247e-05, 1.73194623e-04, 2.94960745e-04,
               3.90257203e-04, 4.51127820e-05, 0.00000000e+00, 4.51127820e-05,
               9.19078182e-04, 1.05223530e-03, 1.05745115e-03, 1.15549920e-03,
               4.51127820e-05, 0.00000000e+00, 8.12661454e-04, 3.00919222e-04,
               6.90185057e-04, 4.79207921e-04, 1.01730513e-03, 3.80202475e-04,
               2.01530997e-04, 5.78486433e-04, 0.00000000e+00, 1.94449340e-03,
               4.51127820e-05, 3.80202475e-04, 2.56862745e-04, 9.76465159e-04,
               8.35272815e-04, 6.54266220e-04, 2.78260870e-04, 1.02170561e-03,
               1.10662443e-03, 0.00000000e+00, 1.31671622e-03, 4.87804878e-05,
               0.00000000e+00, 4.88232580e-04, 3.75469337e-05, 4.97849534e-04,
               4.79207921e-04, 0.00000000e+00, 2.61437060e-04, 6.83537798e-04,
               2.66696443e-04, 2.91316527e-04, 2.17098004e-03, 8.22228774e-04,
               6.27898893e-03, 2.16485556e-03, 3.13462911e-03, 4.00771962e-03,
               3.71332587e-03, 1.98871658e-03, 5.67779189e-03, 1.28000000e-03,
               4.17085785e-03, 9.60465101e-03, 1.36981267e-03, 3.03326141e-03,
               3.02933257e-03, 1.32076161e-03, 1.18466043e-03, 9.52013370e-04,
               1.24505543e-03, 4.00000000e-04, 1.99284077e-03, 8.26587942e-04,
               1.18972228e-03, 8.27963526e-04, 2.07620682e-03, 6.75862069e-04,
               1.31282144e-03, 1.72150693e-03, 1.61940360e-03, 2.20816809e-03,
               1.72330973e-03, 1.01716185e-03, 3.01897302e-03, 2.19893573e-03,
               1.87133277e-03, 3.12756283e-03, 1.40419874e-03, 2.18299805e-04,
               5.41326041e-03, 0.00000000e+00, 4.34059966e-03, 1.68516713e-03,
               6.42730662e-03, 4.62007157e-03, 4.79393385e-03, 3.60758630e-03,
               2.89883769e-03, 2.32716649e-03, 4.22484934e-03, 6.93417871e-03,
               2.09523810e-03, 5.81761290e-03, 9.97172159e-03, 5.38589556e-03,
               5.19147857e-04, 6.17994835e-03, 2.59936311e-03, 3.49717560e-03,
               3.02848058e-03, 2.69677419e-03, 2.77962747e-03, 2.12456361e-03,
               9.08492245e-03, 5.17233298e-03, 7.31500686e-03, 9.78023501e-03,
               4.30802551e-03, 6.42199544e-03, 1.75157420e-03, 2.84767164e-03,
               4.52791564e-03, 3.92448456e-03, 7.08134748e-03, 5.73508410e-03,
               2.75130354e-03, 9.65052352e-03, 4.03520095e-03, 6.29717560e-03,
               3.67408014e-03, 7.13270936e-03, 2.00000000e-03, 5.19014970e-03,
               7.17240498e-03, 1.83559222e-04, 9.08343354e-03, 5.86913688e-03,
               7.98872361e-03, 6.59340659e-05, 0.00000000e+00, 4.63068732e-03,
               3.74902398e-03, 5.96078431e-03, 1.65438683e-03, 1.71307539e-03,
               9.25919920e-03, 2.15615963e-03, 6.96912573e-03, 8.91847611e-03,
               5.87913305e-03, 5.50186800e-03, 1.91796054e-03, 1.24749758e-02,
               2.97575083e-03, 5.67295278e-03, 9.42477321e-03, 7.09238347e-03,
               4.25097602e-03, 5.49579074e-03, 3.66665497e-03, 2.00000000e-03,
               5.70005047e-03, 4.66501216e-03, 4.02248169e-03, 1.13567347e-02,
               7.52795087e-03, 3.28353564e-03, 1.45338651e-02, 3.91561181e-03,
               9.51540934e-03, 5.47067153e-03, 3.99239478e-03, 4.05429321e-03,
               7.71307129e-03, 5.91458201e-03, 1.04464685e-03, 3.43343109e-03,
               7.43071265e-03, 1.47743331e-03, 1.78711485e-03, 5.23791816e-03,
               7.65513020e-03, 9.24380915e-03, 1.52225694e-02, 8.85259318e-03,
               6.92605715e-03, 1.34326858e-02, 3.99722607e-03, 6.72603772e-03,
               1.05504926e-02, 5.67627683e-03, 3.84210526e-03, 1.13134044e-02,
               8.04205525e-03, 3.98000000e-03, 1.11511085e-02, 1.50021041e-02,
               2.56241178e-03, 3.82588478e-03, 3.74902398e-03, 2.00000000e-03,
               1.71626767e-02, 1.25663027e-02, 7.55224187e-03, 4.22409474e-03,
               4.89372697e-03, 7.71902346e-03, 8.56655942e-03, 5.93964051e-03,
               4.98439418e-03, 5.49818067e-03, 8.00865898e-03, 5.23655821e-03,
               1.70282440e-03, 4.70728216e-03, 1.96656834e-03, 6.49934771e-03,
               1.18426739e-02, 4.50062913e-03, 8.99417574e-03, 1.19464715e-02,
               7.80594898e-03, 3.80492434e-03, 9.52380952e-05, 5.90420288e-03,
               5.09766595e-03, 7.48939576e-03, 3.71367431e-03, 1.06686780e-02,
               9.00273716e-03, 2.00000000e-03, 8.90910722e-03, 7.32562370e-03,
               6.05308086e-03, 1.17319447e-02, 5.74091526e-03, 3.74149660e-03,
               5.73513020e-03, 1.70282440e-03, 3.68196509e-03, 1.17672007e-02,
               2.03568761e-03, 5.70582617e-03, 1.53684783e-03, 5.85523979e-03]
ada_features = [0.11, 0.01, 0.00, 0., 0., 0.006, 0.004, 0.028, 0.02, 0.004, 0.006,
                0.002, 0.004, 0.006, 0., 0.004, 0.004, 0., 0., 0., 0.004, 0.,
                0.008, 0.002, 0.008, 0.002, 0.004, 0.004, 0., 0., 0.008, 0.006,
                0.004, 0.002, 0.002, 0.002, 0.002, 0.056, 0.004, 0.002, 0.006,
                0.004, 0.008, 0.002, 0.002, 0.002, 0., 0., 0., 0.004, 0.006, 0.008,
                0.004, 0.002, 0.002, 0.004, 0.008, 0.018, 0.022, 0.014, 0.02,
                0.014, 0.016, 0.01,  0., 0.008, 0.002, 0.008, 0., 0.008, 0., 0.006,
                0.006, 0.008, 0., 0.002, 0., 0., 0., 0.042, 0.034, 0., 0., 0.004,
                0.002, 0.006, 0.008, 0.006, 0., 0., 0., 0.002, 0.004, 0.002, 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.024, 0., 0., 0.,
                0.002, 0., 0., 0., 0.056, 0.006, 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.006, 0.012, 0., 0.,
                0., 0., 0.002, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.002, 0., 0., 0., 0.002, 0.002, 0., 0., 0.002, 0., 0.002,
                0.004, 0., 0.01, 0.016, 0.006, 0., 0., 0., 0.01, 0., 0.002, 0.,
                0.002, 0., 0., 0., 0., 0., 0., 0.006, 0.014, 0.022, 0.008, 0.008,
                0.008, 0., 0.002, 0., 0.004, 0., 0.002, 0., 0.006, 0.004, 0.002,
                0., 0.01, 0.002, 0.004, 0., 0., 0.002, 0., 0.002, 0.002, 0., 0.,
                0.01, 0., 0., 0.006, 0., 0., 0.006, 0., 0., 0.002, 0.006, 0.,
                0.002, 0., 0., 0.004, 0., 0., 0., 0.014, 0.002, 0., 0., 0., 0.004, 0.]
gb_features = [1.37202317e-02, 2.25172021e-03, 7.24297534e-03, 3.45468580e-03,
               4.65228258e-03, 2.89852797e-03, 5.78329610e-03, 5.58721671e-03,
               4.35215231e-03, 1.64947571e-03, 6.75592473e-03, 5.28800274e-03,
               5.33831757e-04, 5.09632177e-03, 3.65129065e-03, 1.81275208e-03,
               3.24568044e-04, 1.58724198e-03, 2.58613069e-03, 4.62047472e-03,
               2.14476679e-03, 2.45963571e-03, 0.00000000e+00, 4.10419344e-03,
               2.79049431e-03, 1.75181213e-03, 6.31424846e-03, 2.86074341e-03,
               6.91383868e-03, 1.94892922e-03, 4.68673473e-03, 6.44262846e-03,
               8.53236169e-03, 1.58821404e-03, 1.80793135e-03, 6.46286319e-03,
               1.67208428e-03, 1.71046129e-03, 5.59592263e-03, 4.60021161e-03,
               3.37709135e-03, 3.02953371e-03, 1.07567948e-02, 7.52174255e-03,
               2.70005383e-03, 7.12029125e-04, 8.52742075e-04, 2.31247013e-03,
               7.21451171e-04, 3.52991470e-03, 5.02645814e-03, 1.98329065e-03,
               1.92201107e-03, 5.23425646e-03, 5.32223150e-03, 3.67993238e-03,
               2.74416732e-03, 2.38800857e-03, 2.91789983e-03, 3.41454894e-03,
               4.76230437e-03, 4.83583803e-03, 7.16713748e-03, 4.78792353e-03,
               6.13765078e-03, 6.58822239e-03, 4.02743090e-03, 1.14024817e-02,
               1.17961637e-02, 8.62175694e-03, 8.40735965e-03, 4.97554402e-03,
               1.00056060e-02, 2.92613562e-03, 4.46866996e-03, 7.41751560e-03,
               5.40931073e-03, 6.49422161e-03, 5.08594847e-03, 1.11636065e-02,
               1.00135952e-02, 1.14165180e-03, 0.00000000e+00, 1.77734256e-03,
               2.11249197e-03, 4.07959567e-03, 2.94604454e-03, 1.54723728e-03,
               8.02511004e-04, 5.93263698e-04, 2.02448433e-03, 2.05993259e-03,
               2.20673386e-03, 1.97901708e-03, 0.00000000e+00, 0.00000000e+00,
               8.70794901e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               1.01227645e-03, 0.00000000e+00, 0.00000000e+00, 9.15160485e-04,
               9.94340469e-04, 1.87456526e-03, 2.00000000e-03, 6.88830583e-04,
               4.73549229e-05, 0.00000000e+00, 5.52624199e-04, 0.00000000e+00,
               7.44131113e-05, 0.00000000e+00, 9.38235934e-03, 4.51748119e-04,
               0.00000000e+00, 0.00000000e+00, 1.99845181e-03, 0.00000000e+00,
               3.59379793e-04, 0.00000000e+00, 0.00000000e+00, 6.17158186e-04,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.79705467e-03,
               5.55804890e-04, 0.00000000e+00, 0.00000000e+00, 1.55279238e-03,
               0.00000000e+00, 5.18075311e-04, 0.00000000e+00, 9.41968036e-04,
               7.83990239e-04, 0.00000000e+00, 8.00257961e-04, 0.00000000e+00,
               0.00000000e+00, 6.90777368e-04, 0.00000000e+00, 0.00000000e+00,
               9.16720253e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               2.12460458e-04, 3.99609707e-04, 1.45419497e-03, 5.34224909e-04,
               5.29412781e-04, 4.41416082e-04, 8.38196861e-04, 1.23537056e-03,
               6.57640470e-04, 0.00000000e+00, 7.52476618e-04, 1.31428014e-03,
               0.00000000e+00, 6.44982125e-04, 4.81941409e-03, 1.57468896e-03,
               2.29679368e-03, 5.04358563e-03, 3.95647765e-03, 5.90854575e-03,
               1.31447615e-03, 0.00000000e+00, 0.00000000e+00, 2.99475223e-03,
               1.58188490e-03, 3.31530464e-04, 0.00000000e+00, 0.00000000e+00,
               1.91948644e-03, 3.02770644e-04, 1.51833914e-03, 1.31957665e-03,
               4.36221912e-04, 3.93651671e-04, 4.54434642e-04, 1.77210051e-03,
               6.01909441e-04, 4.24628154e-04, 8.39129965e-04, 5.06481069e-03,
               0.00000000e+00, 0.00000000e+00, 6.78947898e-04, 2.24045762e-03,
               4.44367434e-03, 5.77660933e-03, 2.63622077e-03, 2.27779321e-03,
               4.89123875e-03, 4.82670151e-03, 0.00000000e+00, 1.93031558e-03,
               0.00000000e+00, 8.83168788e-04, 7.88293805e-04, 5.23016349e-05,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.99283750e-03,
               1.38749297e-03, 1.04934826e-03, 2.64440304e-03, 4.38060352e-04,
               1.26470047e-03, 5.22009426e-03, 7.01992362e-04, 1.00433204e-03,
               3.52247156e-04, 8.47414778e-04, 0.00000000e+00, 0.00000000e+00,
               1.00585267e-03, 1.18833556e-03, 1.04509468e-04, 1.21502348e-03,
               1.93472421e-03, 1.63041977e-03, 1.71093462e-04, 0.00000000e+00,
               1.02848730e-03, 2.07509556e-03, 2.15120063e-03, 7.37039291e-04,
               3.53871685e-04, 6.46789924e-04, 0.00000000e+00, 0.00000000e+00]

cols = train.columns.values

feature_dataframe = pd.DataFrame({
    'features': cols,
    'rf feature importances': rf_feature,
    'et feature importances': et_feature,
    'ada feature importances': ada_feature,
    'gb feature importances': gb_feature,
})


def trace_plot(importances):
    trace = go.Scatter(
        y=feature_dataframe[importances].values,
        x=feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            # size=25,
            size=feature_dataframe['ada feature importances'].values,
            # color=np.random.randn(500),
            color=feature_dataframe[importances].values,
            colorscale='Portland',
            showscale=True,
        ),
        text=feature_dataframe['features'].values
    )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        title=importances,
        hovermode='closest',
        xaxis=dict(
            title='Geno',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    plty.iplot(fig, filename=importances)

trace_plot('rf feature importances')
trace_plot('et feature importances')
trace_plot('ada feature importances')
trace_plot('gb feature importances')

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)
feature_dataframe.head(3)

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
    x=x,
    y=y,
    width=0.5,
    marker=dict(
        color=feature_dataframe['mean'].values,
        colorscale='Portland',
        showscale=True,
        reversescale=False,
    ),
    opacity=0.6
    )]

layout = go.Layout(
    autosize=True,
    title='Barplots of Mean Feature Importance',
    hovermode='closest',
    xaxis=dict(
        title='Pop',
        ticklen=5,
        zeroline=False,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
plty.iplot(fig, filename='bar-direct-labels')

# with kfold
base_predictions_train = pd.DataFrame({
    'RandomeForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel(),
})
base_predictions_train.head()

data = [
    go.Heatmap(
        z=base_predictions_train.astype(float).corr().values,
        x=base_predictions_train.columns.values,
        y=base_predictions_train.columns.values,
        colorscale='Portland',
        showscale=True,
        # reverscale=True
    )
]

plty.iplot(data, filename='labelled-heatmap')

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
    'Density': Density,
    'DensPreds': predictions,
})
StackSub.to_csv("../Output/StackSub.csv", index=False)

print(predictions)


def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
best_params, best_score, score_results = maximize(func, param_grid, args, verbose=False)

print(best_params, best_score, score_results)
