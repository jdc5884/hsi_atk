# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

"""
For testing correlation of particular wavelengths with specific labels.
Outputs CSV file with accuracy scores are particular indices (wavelengths)
to associate with desired labels for prediction.
"""


import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import svm

# Read in data to pandas csv object
hyper_data = pd.read_csv('../Data/headers3mgperml.csv', sep=',')
hyper_datar = pd.read_csv('../Data/massaged_data.csv', sep=',')
headers = hyper_data.columns


relevant_scores = open('~/PycharmProjects/HyperSpectralImaging/Output/relevant_scores.csv', 'w')

relevant_writer = csv.writer(relevant_scores, delimiter=',')
relevant_writer.writerow(['wavelength0', 'wavelength1', 'dtc', 'svc', 'dtr'])             # writing headers

for i in range(15, 254, 2):
    # Specifying data index slices
    X = hyper_data.values[:, i:i+2]        # Signal data from specific wavelengths
    Y = hyper_data.values[:, 3]            # Label data from Nitrogen values

    # Separating training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    # Creating DecisionTree Regressor
    clf_dtc = DecisionTreeClassifier()
    clf_dtc.fit(X_train, y_train)           # Training Decision Tree regressor

    # Creating SVC SVM
    clf_svc = svm.SVC()
    clf_svc.fit(X_train, y_train)           # Training SVR SVM

    y_pred_dtc = clf_dtc.predict(X_test)
    y_pred_svc = clf_svc.predict(X_test)

    dtc_accuracy = accuracy_score(y_test, y_pred_dtc)
    svc_accuracy = accuracy_score(y_test, y_pred_svc)

    Xr = hyper_datar.values[:, i-2:i]
    Yr = hyper_datar.values[:, 3]

    X_trainr, X_testr, y_trainr, y_testr = train_test_split(Xr, Yr)

    clf_dtr = DecisionTreeRegressor()
    clf_dtr.fit(X_trainr, y_trainr)

    #clf_svr = svm.SVR()
    #clf_svr.fit(X_trainr, y_trainr)

    #clf_tsr = TheilSenRegressor(random_state=100)
    #clf_tsr.fit(X_trainr, y_trainr)

    y_pred_dtr = clf_dtr.predict(X_testr)
    #y_pred_svr = clf_svr.predict(X_testr)
    #y_pred_tsr = clf_tsr.predict(X_testr)

    dtr_accuracy = accuracy_score(y_testr, y_pred_dtr)
    #svr_accuracy = accuracy_score(y_testr, y_pred_svr)
    #tsr_accuracy = accuracy_score(y_testr, y_pred_tsr)

    relevant_writer.writerow([headers[i],headers[i+1], dtc_accuracy, svc_accuracy, dtr_accuracy])