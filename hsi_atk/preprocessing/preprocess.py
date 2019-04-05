import numpy as np
from mlens.externals.sklearn.base import BaseEstimator, TransformerMixin

__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"


class SliceNDice(BaseEstimator, TransformerMixin):

    def __init__(self, slice=None):
        self.slice = slice  # n-tuple determining what dim to slice to what val (n-dim, idx)
                            # for now we will just do int for preselected dim

    def fit(self, X, y=None):  #TODO: use_loc_?
        return self

    def transform(self, X, y=None, copy=False):

        # shape = X.shape if X.isinstance(np.ndarray) else None

        if self.slice is None:
            return X

        else:
            Xt = X.copy if copy else X

            Xt = Xt[:,self.slice,:]

        return Xt