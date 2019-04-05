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

def randomImage(n_histograms, n_bins, inf_set):
    mus = np.random.rand(n_histograms) * 10 + 3
    std = np.random.rand(n_histograms) * 2 + 1
    hs = [randNormal(n_bins, mus[i], std[i]) for i in range(n_histograms)]
    i, j, k = inf_set
    labels = [f1(mus[i]), f2(mus[j]), f3(mus[k])]
    return hs, labels


def randomProblem(n_images, n_histograms, n_bins, n_informative=3):
    inf_set = np.linspace(0, n_histograms-1, n_histograms).astype(np.int)
    # print(inf_set)
    np.random.shuffle(inf_set)  #TODO: enforce n_histograms >= n_informative
    inf = inf_set[0:n_informative]
    # print(inf)
    X = []
    V = []

    for l in range(n_images):
        hs, labels = randomImage(n_histograms, n_bins, inf)
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
