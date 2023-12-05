import torch 
# from CWT_CNN import ConvNet
from static_CNN import ConvNet
# from log_print import ParallelLayersModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state_dict = torch.load("./model112.pth")
model = ConvNet()
# model=ParallelLayersModel()
# model.load_state_dict(state_dict)
model.to(device)
model.eval()
dummy_input = torch.randn(1,1,368).cuda()
# model(dummy_input)
# torch.onnx.dynamo_export(model, dummy_input)
torch.onnx.export(model, dummy_input,'model.onnx')
#  input_names = ['input'], output_names = ['output'])
