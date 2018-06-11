import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from HSI_ATK.generators.simple_gen import add_noise_2d


seed = 2018
np.random.seed(seed)

image_set = np.genfromtxt('../testdata/c1_gn.csv', delimiter=',')
image_set = add_noise_2d(image_set)
l_space = np.genfromtxt('../testdata/c1_xy.csv', delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(image_set, l_space, test_size=.12)

rfc = RandomForestClassifier(random_state=seed)
gbc = GradientBoostingClassifier(random_state=seed)

rfc.fit(X_train, y_train)
gbc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
gbc_pred = gbc.predict(X_test)

print(accuracy_score(y_test, rfc_pred))
print(accuracy_score(y_test, rfc_pred))
