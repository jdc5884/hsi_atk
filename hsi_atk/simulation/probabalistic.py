import numpy as np


class Probabalistic:

    def __init__(self, n_samples=100, pix=66000, bands=240, reflectance=4096, centers=8,
                 center_probs=(0, .0625, .125, .375, .125, .0625, .125, .125),
                 center_bands=(30, 50, 75, 100, 120, 150, 200, 220),       # Generate random options for n_centers with a
                 center_mus=(0, 300, 300, 400, 500, 500, 600, 1500, 2000), # defined min/max integer range for each the
                 center_stds=(1, 50, 50, 50, 100, 50, 300, 500, 500),         # center_bands, center_mus, and center_stds
                 classes=(2, 2, 2, 6), class_cor=(50, 100, 200, 220),
                 continuous=(1, 1, 1, 1, 1, 1), continuous_range=(()), continuous_cor = (50, 200),
                 type="histograms", bins=240):
        # set nclasses and ncontinuous to allow for easier covariance assignment

        self.n_samples = n_samples
        self.pix = pix
        self.bands = bands
        self.reflectance = reflectance

        self.centers = centers
        self.center_probs = center_probs
        self.center_bands = center_bands      # may not need
        self.center_mus = center_mus
        self.center_stds = center_stds

        self.classes = classes
        self.class_cor = class_cor            # bandwise correlation to class
        self.continuous = continuous
        self.continuous_range = continuous_range
        self.continuous_cor = continuous_cor  # bandwise correlation to cont. variable

        self.type = type
        self.bins = bins

        # self.generateProblem()

    def generateProblem(self):
        # hists = []
        mus = np.zeros(self.bins)
        stds = np.zeros((self.bins, self.bins))
        bin_div = int(self.bins/self.centers)
        for i in range(self.centers):
            # center_mus has centers+1 values to add zero
            mu1 = self.center_mus[i]
            mu2 = self.center_mus[i+1]
            mus[i*bin_div:(i+1)*bin_div] += np.linspace(mu1, mu2, bin_div)
            std1 = self.center_stds[i]
            std2 = self.center_stds[i+1]
            rr = np.linspace(i*bin_div, (i+1)*bin_div-1, bin_div).astype(int)
            stds[rr,rr] += np.linspace(std1, std2, bin_div)

        shape = (self.n_samples, self.pix, self.bands, self.bins)
        hists = np.random.multivariate_normal(mus, stds, shape)
        return hists



def makeHist(bins, mu, std):
    pass

def makeLabel(hists):
    pass
