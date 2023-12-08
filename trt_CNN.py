import argparse
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
label=['F', 'N', 'Q','V']
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbose output",
)
parser.add_argument(
    "-u",
    "--url",
    type=str,
    required=False,
    default="140.116.233.113:8001",
    help="Inference server URL. Default is localhost:8001.",
)
parser.add_argument(
    "-t",
    "--client-timeout",
    type=float,
    required=False,
    default=None,
    help="Client timeout in seconds. Default is None.",
)

FLAGS = parser.parse_args()
try:
    triton_client = grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    )
except Exception as e:
    print("context creation failed: " + str(e))
    sys.exit()

FLAGS = parser.parse_args()
try:
    triton_client = grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    )
except Exception as e:
    print("context creation failed: " + str(e))
    sys.exit()

model_name = "CWT_CNN"
# Infer
inputs = []
outputs = []
inputs.append(grpcclient.InferInput("input.1", [1,1, 368], "FP32"))
outputs.append(grpcclient.InferRequestedOutput("280"))

def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


def CNN_processing(seg_list):
    pred_list=[]
    user_data = []
    for i, seg in enumerate(seg_list):
        # 創建 PILLOW 
        # image = Image.fromarray((image * 255).astype(np.uint8))
        # input_data = transform(image).unsqueeze(0).to(device)
        seg=np.float32(seg)
        seg = seg[None,None,:]
        
        num_examples  = seg.shape[0]
        signal_length = seg.shape[-1]
        # 進行推理
        inputs[0].set_data_from_numpy(seg)
        
        triton_client.async_infer(
        model_name=model_name,
        inputs=inputs,
        callback=partial(callback, user_data),
        outputs=outputs,
        client_timeout=FLAGS.client_timeout)
    time_out = 1000
    while (len(user_data) == 0) and time_out > 0:
        time_out = time_out - 1
        time.sleep(0.001)
    # Display and validate the available results
    if len(user_data) >=1:
        # Check for the errors
        if type(user_data[0]) == InferenceServerException:
            print(user_data[0])
            sys.exit(1)
        # Validate the values by matching with already computed expected
        # values.
        for ind in range(len(user_data)):
            output=user_data[ind].as_numpy("280")
            #印出結果
            pred = np.argmax(output)
            label_index=pred.item()
            pred_list.append(label[label_index])
              
    return pred_list
