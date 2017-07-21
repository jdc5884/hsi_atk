__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#from HyperspecClassifier import *

# Read in data to pandas csv object
hyper_data = pd.read_csv('headers3mgperml.csv', sep=',')

# Specifying data index slices
X = hyper_data.values[:, 15:]       # Signal data from specific wavelengths
Y = hyper_data.values[:, 3]         # Label data from Density values

# Separating training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, )

# X_train[:, 121:123][1:] *= 5      # Slices at wavelengths that may correlate to nitrogen
# X_train[:, 144][1:] *= 5          # presence
# X_train[:, 185][1:] *= 15

ran_F = RandomForestClassifier()    # Create classifier
ran_F.fit(X_train, y_train)         # Train to data
y_pred = ran_F.predict(X_test)      # Create prediction set

ext_T = ExtraTreesClassifier()      # Create classifier
ext_T.fit(X_train, y_train)         # Train to data
y_pred_et = ext_T.predict(X_test)   # Create prediction set

print(y_pred, '\n', y_pred_et)      # Printing prediction sets
                                    # and accuracy scores as percents
print(accuracy_score(y_test, y_pred)*100, '\n', accuracy_score(y_test, y_pred_et)*100)