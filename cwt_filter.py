import six
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch.autograd import Variable


def filter_build(dt,unbias=False):
    w=6
    coeff = np.sqrt(w* w + 2)
    scales=(np.reciprocal(np.arange(40,4,-36/112))*(coeff+w))/(4.*np.pi)
    filters = [None]*len(scales)
    for scale_idx, scale in enumerate(scales):
        # Number of points needed to capture wavelet
        M = 10 * scale / dt
        # Times to use, centred at zero
        t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
        if len(t) % 2 == 0: t = t[0:-1]  # requires odd filter size
        # Sample wavelet and normalise
        norm = (dt / scale) ** .5

        x = t / scale
        wavelet = np.exp(1j * w * x)
        wavelet -= np.exp(-0.5 * (w ** 2))
        wavelet *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)
        filters[scale_idx] = norm * wavelet
    
    return filters



