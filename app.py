# main.py
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from segment import segment
# from CWT_CNN import CNN_processing
from CWT112 import CWT_112
from trt_CNN import CNN_processing as CNN_processing2
from CNN import CNN_processing as CNN_processing1
import json
import numpy as np
from scipy.signal import butter, lfilter
from tfa_morlet_112m import filter_fir1
import httpx
import mysql.connector
import asyncio
import time

app = FastAPI()
clients = []  # 儲存 WebSocket 客户端連接
buffer = []
fs = 512  # 採樣頻率500Hz
segnumber = 0
mydb = mysql.connector.connect(
  host="140.116.233.113",
  user="wtmh",
  password="wtmh0306",
  database="ECG_result"
)
db_config = {
    "host": "140.116.233.113",
    "user": "wtmh",
    "password": "wtmh0306",
    "database" : "ECG_result"
}
mycursor = mydb.cursor()

@app.post("/submit")
async def receive_data(request: Request):
    global buffer
    global data
    global segnumber
    seg_list=[]
    prediction_list=[]
    rpeak_list=[]
    rpeak_list_j = json.dumps(rpeak_list)
    data_request = await request.json()  # data
    data = buffer+ data_request

    # start_time = time.time()
    data = filter_fir1(data)
    # end_time = time.time()
    # print(f"filter Execution time: {end_time - start_time} seconds")

    data = data.tolist()
    # start_time = time.time()
    rpeaks = segment(data)
    for i in range(len(rpeaks)):
            if rpeaks[i]-184<0:
                print("discard")
            elif rpeaks[i]+184>512:
                start = rpeaks[i]-184
                end = 512
                buffer = data[start:end]
            elif rpeaks[i]-184>=0 and rpeaks[i]+184<=512:
                for i in range(len(rpeaks)):
                    seg_start = rpeaks[i] - 184  # 設定每個R波峰值附近的心電圖訊號段的起點
                    seg_end = rpeaks[i] + 184  # 設定每個R波峰值附近的心電圖訊號段的終點
                    seg_one = data[seg_start:seg_end]  # 切割心電圖訊號
                
                if len(seg_one) == 368:
                    seg_list.append(seg_one)
                    rpeak_list.append(int(rpeaks[i]))
    # end_time = time.time()
    # print(f"segment Execution time: {end_time - start_time} seconds")

    #####################################################################################################################
    #CWT & CNN
    
    if len(seg_list)!=0:
        # start_time = time.time()
        prediction_list = CNN_processing2(seg_list)
        # end_time = time.time()
        # print(prediction_list)
        # print(f"CNN+CWT Execution time: {end_time - start_time} seconds")

        pred_list_j = json.dumps(prediction_list)
        seg_list_j = json.dumps(seg_list)
    else:
        pred_list_j=json.dumps(prediction_list)
        seg_list_j = json.dumps(seg_list)


    # if len(seg_list)!=0:
    #     start_time = time.time()
    #     CWT_list=CWT_112(seg_list)
    #     end_time = time.time()
    #     print(f"CWT Execution time: {end_time - start_time} seconds")
    #     start_time = time.time()
    #     prediction_list = CNN_processing1(CWT_list)
    #     end_time = time.time()
    #     print(prediction_list)
    #     print(f"CNN Execution time: {end_time - start_time} seconds")

    #     pred_list_j = json.dumps(prediction_list)
    #     seg_list_j = json.dumps(seg_list)
    # else:
    #     pred_list_j=json.dumps(prediction_list)
    #     seg_list_j = json.dumps(seg_list)

    #####################################################################################################################
    for client in clients:
    # 將三個數據集合併為一個字典
        # await client.send_json(data)
        combined_data = {
            "ecg_data": data,
            "rpeak_list": rpeak_list_j,
            "predictions": pred_list_j,
        }
        await client.send_json(combined_data)
        # 向所有用戶端傳輸數據
    # start_time = time.time()
    if(len(prediction_list) != 0):
        for i in range(len(seg_list)):
            seg_data = seg_list[i]
            predictions = prediction_list[i]
            seg_data = json.dumps(seg_data)
           
            sql = "INSERT INTO ecg_datas (segnumber, seg_data, Prediction) VALUES (%s, %s, %s)"
            val = [
            (segnumber, seg_data, predictions),
            ]
            mycursor.executemany(sql, val)
            mydb.commit()
            segnumber += 1
            print(mycursor.rowcount, "was inserted.")
    # end_time = time.time()
    # print(f"SQL Execution time: {end_time - start_time} seconds")


    return {"Received": combined_data}     
    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            _ = await websocket.receive_text()  # 保持連接
    except Exception as e:
        clients.remove(websocket)

@app.websocket("/ws1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            with mysql.connector.connect(**db_config) as db:
                with db.cursor(dictionary=True) as cursor:
                    cursor.execute("SELECT * FROM ecg_datas")
                    data = cursor.fetchall()
                    await websocket.send_json(data)
        except mysql.connector.Error as e:
            print("Database error:", e)
        await asyncio.sleep(1)  # 每秒查訪一次SQL

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("./data_display.html",encoding='utf-8') as f:
        Ht=f.read()
    return Ht


@app.post("/clear-table")
async def clear_table():
    try:
        mycursor = mydb.cursor()
        mycursor.execute("TRUNCATE TABLE ecg_datas")
    except mysql.connector.Error as e:
        print("Database error:", e)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
        