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
y_train = train.values[:, 2]
X_test = test.values[:, 13:]

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

# TODO: put feature to python lists for weighting
# TODO: may be able to use tree attribute/method 'feature_importances_'
# TODO: which returns arrays with class weights

rf_features = []
et_features = []
ada_features = []
gb_features = []

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
            size=25,
            # size = feature_dataframe['ada feature importances'].values,
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
        # xaxis=dict(
        #     title='Geno',
        #     ticklen=5,
        #     zeroline=False,
        #     gridwidth=2,
        # ),
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
    'GenoTypes': GenoTypes,
    'GenoPreds': predictions,
})
StackSub.to_csv("../Output/StackSub.csv", index=False)

print(predictions)


def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
best_params, best_score, score_results = maximize(func, param_grid, args, verbose=False)

print(best_params, best_score, score_results)
