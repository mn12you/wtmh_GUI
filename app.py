from flask import Flask, render_template, request
import os
from segment import segment
from CNN import CNN_processing
from CWT112 import CWT_112
import json
import numpy as np

app = Flask(__name__)

# 設置上傳文件的目錄
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 確保目錄存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        cwt_list = []
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        seg_list = segment(file_path)
        seg_list = np.array(seg_list)
        seg_list = seg_list.tolist()
        seg_list_j = json.dumps(seg_list)
        for i in range(len(seg_list)):
            CWT = CWT_112(seg_list[i])
            cwt_list.append(CWT)
        cwt_list_np = np.array(cwt_list)
        cwt_list_list = cwt_list_np.tolist()
        cwt_list_j = json.dumps(cwt_list_list)

        pred_list = CNN_processing(cwt_list)
        pred_list_j=json.dumps(pred_list)
        
        return render_template('result.html', file_path=file_path, pred_list=pred_list_j,seg_list=seg_list_j,cwt_list=cwt_list_j)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
