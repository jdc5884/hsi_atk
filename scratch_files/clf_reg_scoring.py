import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as dec
import sklearn.ensemble as en
import sklearn.svm as svm
import sklearn.linear_model as lm

from sklearn.model_selection import train_test_split
import sklearn.metrics as m

pca = dec.PCA(copy=True,whiten=False)

data = pd.read_csv("../Data/headers3mgperml.csv",sep=",")

wavelength= [393.91, 395.982, 398.054, 400.126, 402.198, 404.27, 406.342, 408.414, 410.486, 412.558, 414.63, 416.702, 418.774, 420.846, 422.918, 424.99, 427.062, 429.134, 431.206, 433.278, 435.35, 437.422, 439.494, 441.566, 443.638, 445.71, 447.782, 449.854, 451.926, 453.998, 456.07, 458.142, 460.214, 462.286, 464.358, 466.43, 468.502, 470.574, 472.646, 474.718, 476.79, 478.862, 480.934, 483.006, 485.078, 487.15, 489.222, 491.294, 493.366, 495.438, 497.51, 499.582, 501.654, 503.726, 505.798, 507.87, 509.942, 512.014, 514.086, 516.158, 518.23, 520.302, 522.374, 524.446, 526.518, 528.59, 530.662, 532.734, 534.806, 536.878, 538.95, 541.022, 543.094, 545.166, 547.238, 549.31, 551.382, 553.454, 555.526, 557.598, 559.67, 561.742, 563.814, 565.886, 567.958, 570.03, 572.102, 574.174, 576.246, 578.318, 580.39, 582.462, 584.534, 586.606, 588.678, 590.75, 592.822, 594.894, 596.966, 599.038, 601.11, 603.182, 605.254, 607.326, 609.398, 611.47, 613.542, 615.614, 617.686, 619.758, 621.83, 623.902, 625.974, 628.046, 630.118, 632.19, 634.262, 636.334, 638.406, 640.478, 642.55, 644.622, 646.694, 648.766, 650.838, 652.91, 654.982, 657.054, 659.126, 661.198, 663.27, 665.342, 667.414, 669.486, 671.558, 673.63, 675.702, 677.774, 679.846, 681.918, 683.99, 686.062, 688.134, 690.206, 692.278, 694.35, 696.422, 698.494, 700.566, 702.638, 704.71, 706.782, 708.854, 710.926, 712.998, 715.07, 717.142, 719.214, 721.286, 723.358, 725.43, 727.502, 729.574, 731.646, 733.718, 735.79, 737.862, 739.934, 742.006, 744.078, 746.15, 748.222, 750.294, 752.366, 754.438, 756.51, 758.582, 760.654, 762.726, 764.798, 766.87, 768.942, 771.014, 773.086, 775.158, 777.23, 779.302, 781.374, 783.446, 785.518, 787.59, 789.662, 791.734, 793.806, 795.878, 797.95, 800.022, 802.094, 804.166, 806.238, 808.31, 810.382, 812.454, 814.526, 816.598, 818.67, 820.742, 822.814, 824.886, 826.958, 829.03, 831.102, 833.174, 835.246, 837.318, 839.39, 841.462, 843.534, 845.606, 847.678, 849.75, 851.822, 853.894, 855.966, 858.038, 860.11, 862.182, 864.254, 866.326, 868.398, 870.47, 872.542, 874.614, 876.686, 878.758, 880.83, 882.902, 884.974, 887.046, 889.118]

for wave in range(16,len(wavelength)+16):
    data.values[:,wave] = data.values[:,wave].astype(float)


waveCols = np.array(data.values[:,16:])

genotype = np.array(data.values[:,1])
density = np.array(data.values[:,2])
nitrogen = np.array(data.values[:,3])
palmetic = np.array(data.values[:,10].astype(float))
linoleic = np.array(data.values[:,11].astype(float))
olearic = np.array(data.values[:,12].astype(float))
stearic = np.array(data.values[:,13].astype(float))


labels = [("genotype",genotype),("density",density),
          ("nitrogen",nitrogen),("palmetic",palmetic),
          ("linoleic",linoleic),("olearic",olearic),("stearic",stearic)]

clf = lm.Ridge(normalize=True, solver='sag')

alphas = np.logspace(-10,6,200)

pcs = []
coefs = []
errors = []
scores = []

for n in range(1,241,1):
    pca.set_params(n_components=n)
    pcaS = pca.fit_transform(waveCols)
    X_train, X_test, y_train, y_test = train_test_split(pcaS, palmetic, random_state=1)
    scores.append([clf.set_params(alpha=alpha).fit(X_train,y_train).score(X_test,y_test) for alpha in alphas])

print(max(scores))

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
#
# x_vals = np.repeat(alphas,240)
# y_vals = np.repeat(np.arange(1,241,1),20)
# z_vals = []
# for scoreA in scores:
#     z_vals.extend(scoreA)
#
# X = x_vals
# Y = y_vals
# Z = z_vals
#
# ax.scatter(xs=X, ys=Y, zs=Z, c='red', marker='x')
# ax.set_xlabel('alphas')
# ax.set_ylabel('n_components')
#
# fig.show()