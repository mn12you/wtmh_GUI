from tfa_morlet_112m import tfa_morlet
from tfa_morlet_112m import Cupy_cwt
import numpy as np
import cupy as cp

def CWT_112(seg):
    B = cp.array(seg)
    s_rate = 250

    aa = B - B.mean()
    anorm = aa / aa.max()
    DATA = -anorm
    sig = DATA

    fmin = 4
    fmax = 40
    
    cupy_img = cp.array(Cupy_cwt(sig.get(), s_rate, fmin, fmax, 36/112))
    # tfa_mor_img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112))

    # 確保心電圖至少包含112個資料點
    img_select = cp.linspace(0, 175, 112, dtype=int)
    
    cupy_img = cupy_img[::-1, img_select]
    # tfa_mor_img = tfa_mor_img[::-1, img_select]

    cupy_img = cupy_img / cp.max(cupy_img)
    # tfa_mor_img = tfa_mor_img / np.max(tfa_mor_img)

    cupy_CWT_picture = cupy_img
    # tfa_mor_CWT_picture = tfa_mor_img
    
    # return tfa_mor_img
    return cupy_img