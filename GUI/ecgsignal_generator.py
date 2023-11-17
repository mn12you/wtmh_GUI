import time
with open("D:/lab/AHA Text Data/1001.txt", 'r') as f:
    # 讀取所有行，提取第一個數字
    for line in f.readlines():
        num_list = line.split(',')  #拆分為數字列表
        first_num = int(num_list[0])  #提取第一個元素
        print(first_num)
        time.sleep(1/250)