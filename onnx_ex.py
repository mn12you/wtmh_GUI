import torch 
from CWT_CNN import ConvNet
from cwt_filter import filter_build
Morlet_filter=filter_build(0.004)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("./model112.pth")
model = ConvNet(Morlet_filter)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
dummy_input = torch.randn(1,1,180).cuda()
torch.onnx.export(model, dummy_input, 'model.onnx')
