import biosppy.signals.ecg as ecg
import numpy as np
import matplotlib.pyplot as plt
from CWT112 import CWT_112
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch import nn
import torch.nn.functional as F
from CNN import ConvNet
import time

def CNN_processing(path):
    cwt_images = CWT(path)
    pred_list=[]
    for i, image in enumerate(cwt_images):
    # 創建 PILLOW 
        image = Image.fromarray((image * 255).astype(np.uint8))
        input_data = transform(image).unsqueeze(0).to(device)

        # 進行推理
        with torch.no_grad():
            output = model(input_data)
            pred = output.argmax(dim=1)

        # 列印結果
        a=np.array(['F', 'N', 'Q','V'])
        i=pred.item()
        pred_list.append(a[i])
    return pred_list


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
    img_select = []

    seg_list = segment(path)

    for i in range(0, len(seg_list) - 1):
        seg = seg_list[i]
        CWT_picture=CWT_112(seg)
        CWT_list.append(CWT_picture)

    return CWT_list



if __name__=="__main__":
    
    #讀取 model 放入 GPU 中
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("./model112.pth")
    model = ConvNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #放入模型前處理 (變換大小、灰階、torch tensor)
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])


    file_path="./1001.txt"
    print("Loading...")
    start_time=time.time()
    seg_list = segment(file_path)
    end_time=time.time()
    print("Segmention time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    CWT_list = CWT(file_path)
    end_time=time.time()
    print("CWT time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    predict_list = CNN_processing(file_path)
    end_time=time.time()
    print("CNN time: ",(end_time-start_time))



    # # Create the main GUI window
    # root = tk.Tk()
    # root.title("ECG Segment Viewer")

    # # Create GUI elements
    # segment_label = tk.Label(root, text="ECG Segment 1")
    # segment_label.pack()
    # cwt_label = tk.Label(root, text="label")
    # cwt_label.pack()

    # # Create Matplotlib figures and canvases for displaying ECG and CWT
    # ecg_fig, ecg_plot = plt.subplots(figsize=(6, 4))
    # cwt_fig, cwt_plot = plt.subplots(figsize=(6, 4))
    # ecg_canvas = FigureCanvasTkAgg(ecg_fig, master=root)
    # cwt_canvas = FigureCanvasTkAgg(cwt_fig, master=root)
    # ecg_canvas.get_tk_widget().pack()
    # cwt_canvas.get_tk_widget().pack()

    # # Create previous and next buttons
    # prev_button = tk.Button(root, text="Previous", command=previous_segment)
    # prev_button.pack()
    # next_button = tk.Button(root, text="Next", command=next_segment)
    # next_button.pack()

    # # Create a button to open the file dialog
    # open_button = tk.Button(root, text="Open File", command=open_file_dialog)
    # open_button.pack()

    # root.mainloop()
