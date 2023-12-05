import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(None,'')
with open('model.engine', "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


def TRT_infer(input_image):
    context = engine.create_execution_context()
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            print("A")
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes)
            bindings.append(int(input_memory))
        else:
            print("B")
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    # Synchronize the stream
    stream.synchronize()
    return output_buffer

def CNN_processing(seg_list):
    pred_list=[]
    for i, seg in enumerate(seg_list):
        # 創建 PILLOW 
        # image = Image.fromarray((image * 255).astype(np.uint8))
        # input_data = transform(image).unsqueeze(0).to(device)
        seg = seg[None,None,:]
        # num_examples  = seg.shape[0]
        # signal_length = seg.shape[-1]
        # seg = torch.from_numpy(seg).type(torch.FloatTensor)
        # seg.requires_grad_(requires_grad=False)
        # 進行推理
        with torch.no_grad():
            # seg=seg.cuda()
            output = TRT_infer(seg)
            print(output)
            # pred = output.argmax()

        # 列印結果
        # a=np.array(['F', 'N', 'Q','V'])
        # i=pred.item()
        # pred_list.append(a[i])
    return pred_list

