import numpy as np
# import matplotlib.pyplot as plt


def randNormal(N, mu, std):
    return np.random.normal(loc=mu, scale=std, size=N)

def f1(x):
    return 5*np.exp(-x)

def f2(x):
    return 5*x

def f3(x):
    return 2*np.exp(-2*x)

def randomImage(n, N, inf_set):
    mus = np.random.rand(n) * 10 + 3
    std = np.random.rand(n) * 2 + 1
    hs = [randNormal(N, mus[i], std[i]) for i in range(n)]
    i, j, k = inf_set
    labels = [f1(mus[i]), f2(mus[j]), f3(mus[k])]
    return hs, labels


def randomProblem(n_images, n, N, n_informative=3):
    inf = np.random.randint(0, N, n_informative)
    X = []
    V = []

    for l in range(n_images):
        hs, labels = randomImage(n, N, inf)
        X.append(hs)
        V.append(labels)

    X = np.array(X)
    V = np.array(V)

    return X, V, inf

# X, V, inf = randomProblem(5000, 10, 5, 3)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# mdl = LinearRegression()
#
# mdl.fit(X[:4750,inf[1],:], V[:4750, 1])
# pred = mdl.predict(X[4750:,inf[1],:])
#
# print(mean_squared_error(V[4750:, 1], pred))
