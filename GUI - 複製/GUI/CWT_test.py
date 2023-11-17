from segment import segment
import matplotlib.pyplot as plt
from tfa_morlet_112m import *

def CWT(path):
    CWT_list = []
    selected_data_matrix = []

    seg_list = segment(path)

    for i in range(1, len(seg_list) - 1):
        seg_list = np.array(seg_list)
        
        B = seg_list[i]
        s_rate = 250
        aa = B - B.mean()
        anorm = aa / aa.max()
        B = filter_fir1(anorm)

        aa = B - B.mean()
        anorm = aa / aa.max()
        DATA = -anorm
        sig = DATA

        fmin = 4
        fmax = 40
        img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112, padlen=100))

        # 确保心电图信号段至少包含112个数据点
        if len(seg_list[i]) >= 112:
            # 计算从信号段中选择数据点的间隔
            interval = len(seg_list[i]) // 112
            selected_data = seg_list[i][::interval][:112]
            selected_data_matrix.append(selected_data)

        img = img[::-1, img_select]
        img = img / np.max(img)
        CWT_picture = img
        plt.figure(figsize=(8, 8))
        plt.imshow(CWT_picture, cmap='gray', aspect='auto')
        plt.title(f'CWT Image {i + 1}')
        plt.colorbar()
        plt.xlim(0, 112)
        plt.show()
        CWT_list.append(CWT_picture)

    return CWT_list

CWT("6108.txt")
