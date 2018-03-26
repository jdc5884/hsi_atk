import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from HSI_ATK.Generators.simple_gen import *

image_set, label_set = xy_gen((500,500), 50)
imS = image_set.shape
image_set = image_set.reshape(imS[0], imS[1]*imS[2])

X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, random_state=32, test_size=0.3)

LR = LinearRegression(normalize=True)

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)
acc = accuracy_score(y_pred=y_pred, y_true=y_test)

print(acc)