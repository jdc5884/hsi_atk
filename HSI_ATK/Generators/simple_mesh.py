import numpy as np
import scipy.integrate as int
import scipy.interpolate as ntp
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 1)
y = np.linspace(-5, 5, 1)

xx, yy = np.mgrid()

r = 3
h = 3

def func(x, y, r, h):
    return np.sqrt((x**2 + y**2)/(r/h))

