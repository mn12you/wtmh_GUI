import numpy as np
# from CWT112 import CWT_112
# from CNN import CNN_processing
# from CWT_CNN import CNN_processing
from RT_CNN import CNN_processing
import time
from segment import segment




def CWT(seg_list):
    CWT_list = []
    img_select = []


    for i in range(0, len(seg_list) - 1):
        seg = seg_list[i]
        CWT_picture=CWT_112(seg)
        CWT_list.append(CWT_picture)

    return CWT_list



if __name__=="__main__":

    file_path="./1001.txt"
    print("Loading...")
    start_time=time.time()
    seg_list = segment(file_path)
    end_time=time.time()
    print("Segmention time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    # CWT_list = CWT(seg_list)
    # end_time=time.time()
    # print("CWT time: ",(end_time-start_time))
    # print("Loading...")
    # start_time=time.time()
    predict_list = CNN_processing(seg_list)
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
