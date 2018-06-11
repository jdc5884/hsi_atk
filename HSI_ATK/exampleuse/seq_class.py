import numpy as np
from scipy.stats import uniform, randint

from sklearn.decomposition import RandomizedPCA, PCA, FactorAnalysis, NMF
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifierCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from mlens.ensemble import SequentialEnsemble
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from mlens.preprocessing import Subset


# from HSI_ATK.generators.simple_gen import add_noise_2d
from HSI_ATK.generators.gen3d import silly_gen


seed = np.random.seed(2018)

# image_set = np.genfromtxt('../testdata/c1_gn.csv', delimiter=',')
# label_set = np.genfromtxt('../testdata/c1_lb.csv', delimiter=',')
# l_space = np.genfromtxt('../testdata/c1_xy.csv', delimiter=',')
#
# image_set = add_noise_2d(image_set)
#
# x1train, x1test, y1train, y1test = train_test_split(image_set, label_set, test_size=0.12)
# x2train, x2test, y2train, y2test = train_test_split(image_set, l_space, test_size=0.12)

data_pix, spacial_pix, data, spacial_data = silly_gen(denoise=True)
# mb = MultiLabelBinarizer()
# spacial_pix_L = spacial_pix.astype('int')
# spacial_pix_L = spacial_pix_L.tolist()
# spacial_pix = mb.fit_transform(spacial_pix_L)
indices = np.random.permutation(data_pix.shape[0])
training_idx, test_idx = indices[:1900], indices[1900:]
X_train, X_test = data_pix[training_idx, :], data_pix[test_idx, :]
y_train, y_test = spacial_pix[training_idx], spacial_pix[test_idx]

# X_train, X_test, y_train, y_test = train_test_split(data_pix, spacial_pix, test_size=.23, random_state=seed)

est_l1 = [
    ('etr', ExtraTreesClassifier(n_jobs=1)),
    ('rfr', RandomForestClassifier(n_jobs=1)),
    ('mlp', MLPClassifier(tol=1e-4)),
    ('svc', SVC(tol=1e-4, degree=9)),
    ('rdc', RidgeClassifierCV()),
    ('gbc', GradientBoostingClassifier()),
    ('ada', AdaBoostClassifier()),
    ('svc', SVC(tol=1e-4, degree=7, kernel='linear')),
    ('bag', BaggingClassifier(n_jobs=1))
]

ests_1 = {
    'case-1': est_l1,
    # 'case-2': est_l1,
    # 'case-3': est_l1,
    # 'case-4': est_l1
}

r = uniform(0, 30)
d = randint(2, 10)
f = randint(100, 200)
e = uniform(0, 3)
ee = uniform(0, 1)

pars_1 = {

}

sc = StandardScaler()
pca = PCA()
fa = FactorAnalysis()
nmf = NMF()


pre_cases = {
    'case-1': [sc],
    # 'case-2': [sc],
    # 'case-3': [pca],
    # 'case-4': [fa]
}

score = make_scorer(score_func=accuracy_score, greater_is_better=True, needs_proba=False, needs_threshold=False)

ensemble = SequentialEnsemble(model_selection=True, n_jobs=1, shuffle=False, random_state=seed)

ensemble.add('stack', ests_1, preprocessing=pre_cases)
ensemble.add_meta(SVC(kernel='linear', degree=5, tol=1e-4))
# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
# ens = ensemble
evaluator = Evaluator(scorer=score, random_state=seed, verbose=True)
evaluator.fit(data_pix, spacial_pix, estimators=[], param_dicts=pars_1,
              n_iter=5, preprocessing=pre_cases)

print(evaluator.results)

spacial_pix = spacial_pix.astype('int')
unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)).T)

# print(confusion_matrix(y_test, y_pred, labels=unique))
# print(precision_score(y_test, y_pred, average='micro', labels=unique))
# print(mean_absolute_error(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))

def get_data():
    from HSI_ATK.fileimport import loadImage
    rData = loadImage("../../Data/32.control.bil")
    rData = rData[:, 40:380, 100:530]
    d1, d2, d3 = rData.shape
    rData = rData.swapaxes(0, 2)
    data = rData.transpose(2, 0, 1).reshape(d2*d3, -1)
    return data
