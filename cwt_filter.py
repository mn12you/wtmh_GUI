import six
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.signal
import scipy.optimize
from wavelets import Morlet
import torch
from torch.autograd import Variable
wavelet=Morlet()

def filter_build(dt,unbias=False):
    scales=wavelet.scale_from_period(np.reciprocal(np.arange(40,4,-36/112)))
    filters = [None]*len(scales)
    for scale_idx, scale in enumerate(scales):
        # Number of points needed to capture wavelet
        M = 10 * scale / dt
        # Times to use, centred at zero
        t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
        if len(t) % 2 == 0: t = t[0:-1]  # requires odd filter size
        # Sample wavelet and normalise
        norm = (dt / scale) ** .5
        filters[scale_idx] = norm * wavelet(t, scale)
    
    return filters



