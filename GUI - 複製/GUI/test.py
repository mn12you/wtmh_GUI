import biosppy.signals.ecg as ecg
import numpy as np
import matplotlib.pyplot as plt
from tfa_morlet_112m import *
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
current_index = 0
def segment(path):
    # 讀取txt檔案
    with open(path, 'r') as f:
        # 讀取所有行，提取第一個數字
        numbers = []
        for line in f.readlines():
            num_list = line.split(',')  #拆分為數字列表
            first_num = int(num_list[0])  #提取第一個元素
            numbers.append(first_num)  #添加到數字列表中

    ecg_signal = np.array(numbers)

    # 設定取樣頻率
    fs = 250

    

    # 檢測R波峰值
    rpeaks, = ecg.hamilton_segmenter(ecg_signal, sampling_rate=fs)
    rpeaks, = ecg.correct_rpeaks(ecg_signal, rpeaks, sampling_rate=fs, tol=0.05)

    seg_list = []  # 創建一個空的列表

    for i in range(len(rpeaks)):
        seg_start = rpeaks[i] - 90  # 設定每個R波峰值附近的心電圖訊號段的起點
        seg_end = rpeaks[i] + 90  # 設定每個R波峰值附近的心電圖訊號段的終點
        seg = ecg_signal[seg_start:seg_end]  # 切割心電圖訊號
        if len(seg) == 181:
            seg = np.delete(seg, -1)  # 將多餘的一個數字刪除
        seg_list.append(seg)  # 添加seg到列表中
    
    return  seg_list # 返回包含所有seg的列表

def CWT(path):
    CWT_list = []
    selected_data_matrix = []

    seg_list = segment(path)

    for i in range(1, len(seg_list) - 1):
        DATA = seg_list[i]
        B = np.array([590, 591, 591, 588, 587, 585, 587, 587, 588, 588, 588, 589, 589, 588, 587, 586, 584, 584, 585, 582, 583, 586, 586, 584, 585, 583, 584, 582, 582, 581, 581, 578, 581, 581, 582, 583, 581, 581, 582, 581, 582, 581, 580, 580, 580, 581, 582, 581, 580, 580, 579, 578, 578, 576, 577, 576, 576, 577, 577, 574, 575, 575, 574, 572, 572, 570, 574, 573, 577, 577, 577, 577, 578, 577, 578, 579, 581, 580, 580, 580, 579, 579, 578, 580, 582, 584, 586, 588, 590, 593, 594, 599, 600, 603, 605, 604, 605, 606, 606, 607, 607, 607, 607, 608, 609, 610, 611, 613, 615, 615, 619, 620, 619, 623, 629, 636, 647, 660, 674, 692, 712, 735, 759, 781, 800, 813, 822, 831, 836, 841, 839, 828, 798, 761, 715, 671, 634, 606, 590, 582, 582, 584, 588, 590, 591, 591, 594, 594, 597, 598, 597, 597, 595, 594, 593, 593, 597, 596, 595, 593, 592, 591, 590, 589, 590, 590, 588, 589, 592, 590, 590, 591, 590, 590, 589, 587, 588, 586, 587, 586, 587, 587, 586, 585, 586, 583, 582, 580, 580, 582, 581, 581, 581, 580, 581, 578, 578, 578, 577, 577, 577, 577, 577, 576, 576, 573, 574, 573, 572, 572, 572, 571, 569, 568, 567, 567, 567, 568, 569, 568, 567, 567, 564, 563, 561, 563, 562, 562, 562, 563, 563, 563, 562, 564, 563, 562, 562, 564, 563, 566, 569, 569, 571, 571, 573, 571, 572, 574, 575, 575, 577, 579, 581, 581, 582, 584])
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
        img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112))

        # 确保心电图信号段至少包含112个数据点
        if len(seg_list[i]) >= 112:
            # 计算从信号段中选择数据点的间隔
            interval = len(seg_list[i]) // 112
            selected_data = seg_list[i][::interval][:112]
            selected_data_matrix.append(selected_data)

        img = img[::-1, img_select]
        img = img / np.max(img)
        CWT_picture = img
        
        CWT_list.append(CWT_picture)

    return CWT_list

def show_segment_and_cwt():
    global current_index
    if current_index >= len(seg_list):
        return
    seg = seg_list[current_index]
    cwt_img = CWT_list[current_index]

    # Update the GUI labels and image
    segment_label.config(text=f'ECG Segment {current_index + 1}')
    cwt_label.config(text=f'CWT Image {current_index + 1}')
    ecg_plot.clear()
    cwt_plot.clear()
    ecg_plot.plot(seg, color='blue', label='ECG Segment')
    cwt_plot.imshow(cwt_img, cmap='gray', aspect='auto')
    cwt_plot.set_xlim(0, 112)
    ecg_canvas.draw()
    cwt_canvas.draw()

def previous_segment():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_segment_and_cwt()

def next_segment():
    global current_index
    if current_index < len(seg_list) - 1:
        current_index += 1
        show_segment_and_cwt()

def open_file_dialog():
    global seg_list, CWT_list
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        seg_list = segment(file_path)
        CWT_list = CWT(file_path)
        current_index = 0
        show_segment_and_cwt()

# Create the main GUI window
root = tk.Tk()
root.title("ECG Segment Viewer")

# Create GUI elements
segment_label = tk.Label(root, text="ECG Segment 1")
segment_label.pack()
cwt_label = tk.Label(root, text="CWT Image 1")
cwt_label.pack()

# Create Matplotlib figures and canvases for displaying ECG and CWT
ecg_fig, ecg_plot = plt.subplots(figsize=(6, 4))
cwt_fig, cwt_plot = plt.subplots(figsize=(6, 4))
ecg_canvas = FigureCanvasTkAgg(ecg_fig, master=root)
cwt_canvas = FigureCanvasTkAgg(cwt_fig, master=root)
ecg_canvas.get_tk_widget().pack()
cwt_canvas.get_tk_widget().pack()

# Create previous and next buttons
prev_button = tk.Button(root, text="Previous", command=previous_segment)
prev_button.pack()
next_button = tk.Button(root, text="Next", command=next_segment)
next_button.pack()

# Create a button to open the file dialog
open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack()

root.mainloop()