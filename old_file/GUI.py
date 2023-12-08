import biosppy.signals.ecg as ecg
import numpy as np
import matplotlib.pyplot as plt
from tfa_morlet_112m import *
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=588, out_features=3 * 16)
        self.fc2 = nn.Linear(in_features=3 * 16, out_features=4)

        self.classes = ['F', 'N', 'Q','V']

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x=self.flatten(x)

        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("./model112.pth")
model = ConvNet()
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(112),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

current_index = 0
def CNN(cwt_images):
    
    pred_list=[]
    for i, image in enumerate(cwt_images):
    # 创建Pillow图像对象
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

def CWT(seg_list ):
    CWT_list = []
    img_select = []

    for i in range(0, len(seg_list) - 1):
        seg = seg_list[i]
        B = np.array(seg)
        s_rate = 250
        
        aa = B - B.mean()
        anorm = aa / aa.max()
        DATA = -anorm
        sig = DATA

        fmin = 4
        fmax = 40
        img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112))

        # 确保心电图信号段至少包含112个数据点
        img_select = np.linspace(0, 175, 112, dtype=int)

        img = img[::-1, img_select]
        img = img / np.max(img)
        CWT_picture = img
        
        CWT_list.append(CWT_picture)

    return CWT_list

def CNNprocessing (cwt_images):

    predict_list = CNN(cwt_images)

    return predict_list


def show_segment_and_cwt():

    global current_index

    if current_index >= len(seg_list):
        return
    
    seg = seg_list[current_index]
    cwt_img = CWT_list[current_index]
    predict_label = predict_list[current_index]

    # Update the GUI labels and image
    segment_label.config(text=f'ECG Segment {current_index + 1}')
    cwt_label.config(text=f'CWT Image {predict_label}')
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
    global seg_list, CWT_list, predict_list
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        seg_list = segment(file_path)
        CWT_list = CWT(seg_list)
        predict_list = CNNprocessing (CWT_list)
        current_index = 0
        show_segment_and_cwt()

# Create the main GUI window
root = tk.Tk()
root.title("ECG Segment Viewer")

# Create GUI elements
segment_label = tk.Label(root, text="ECG Segment 1")
segment_label.pack()
cwt_label = tk.Label(root, text="label")
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
