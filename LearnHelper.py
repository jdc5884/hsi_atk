# author David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

class LearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def feature_important(self, X, y):
        print(self.clf.fit(X,y).feature_importances_)
