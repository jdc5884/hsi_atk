

class Pentara_n:

    def __init__(self, name, shape, shape_pars, scale_aoi, scale_fn, label_fns, label_pars):
        base_name = name
        base_shape = shape
        base_params = shape_pars
        sc_aoi = scale_aoi
        sc_fn = scale_fn
        lbl_fns = label_fns
        lbl_pars = label_pars

    def gen_batch(self, n, alpha=0.01, beta=0.1, gamma=0.1, use_std=True, use_mean=True):
        '''Generates batch of pentiga objects
        :param n: int - number of pentigas
        :param alpha: float - range to stretch wavelength parameters
        :param beta: float - range to stretch shape of object
        :param gamma: float - range to correlation coef
        :param use_std: bool - use standard deviation of pixels in label generation
        :param use_mean: bool - use mean of pixels in label generation
        :return: dict of pentiga objects
        '''

        return None
