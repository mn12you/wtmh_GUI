from tfa_morlet_112m import tfa_morlet
import numpy as np
def CWT_112(seg):
    B = np.array(seg)
    s_rate = 250

    aa = B - B.mean()
    anorm = aa / aa.max()
    DATA = -anorm
    sig = DATA

    fmin = 4
    fmax = 40
    img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112))

    # 確保心電圖至少包含112個資料點
    img_select = np.linspace(0, 175, 112, dtype=int)

    img = img[::-1, img_select]
    img = img / np.max(img)
    CWT_picture = img
    return img