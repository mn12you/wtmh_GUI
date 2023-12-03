
import torch
from torchvision.transforms import transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from cwt_filter import filter_build
Morlet_filter=filter_build(0.004)
img_select = np.linspace(0, 179, 112, dtype=int)

class ConvNet(nn.Module):
    def __init__(self,filters):
        super(ConvNet, self).__init__()
        
        self._cuda = True
        self.set_filters(filters)
        
        
        

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

    def set_filters(self, filters, padding_type='SAME'):
        """
        Given a list of temporal 1D filters of variable size, this method creates a
        list of nn.conv1d objects that collectively form the filter bank.

        :param filters: list, collection of filters each a np.ndarray
        :param padding_type: str, should be SAME or VALID
        :return:
        """

        assert isinstance(filters, list)
        assert padding_type in ['SAME', 'VALID']

        self._filters = [None]*len(filters)
        for ind, filt in enumerate(filters):
            
            assert filt.dtype in (np.float32, np.float64, np.complex64, np.complex128)

            if np.iscomplex(filt).any():
                chn_out = 2
                filt_weights = np.asarray([np.real(filt), np.imag(filt)], np.float32)
            else:
                chn_out = 1
                filt_weights = filt.astype(np.float32)[None,:]

            filt_weights = np.expand_dims(filt_weights, 1)  # append chn_in dimension
            filt_size = filt_weights.shape[-1]              # filter length
            padding = self._get_padding(padding_type, filt_size)

            conv = nn.Conv1d(1, chn_out, kernel_size=filt_size, padding=padding, bias=False)
            conv.weight.data = torch.from_numpy(filt_weights)
            conv.weight.requires_grad_(False)

            if self._cuda: conv.cuda()
            self._filters[ind] = conv
    
     
    
    @staticmethod
    def _get_padding(padding_type, kernel_size):
        assert isinstance(kernel_size, int)
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return (kernel_size - 1) // 2
        return 0

    def forward(self, x):
        if not self._filters:
            raise ValueError('PyTorch filters not initialized. Please call set_filters() first.')
            return None
        results = [None]*len(self._filters)
        for ind, conv in enumerate(self._filters):
            results[ind] = conv(x)
        results = torch.stack(results)     # [n_scales,n_batch,2,t]
        cwt = results.permute(1,0,2,3) # [n_batch,n_scales,2,t]
        cwt = (cwt[:,:,0,:] + cwt[:,:,1,:]*1j)
        cwt = cwt[0,0:112, img_select]
        cwt=cwt.resize_(1, 1,112,112)
        cwt=cwt.type(torch.FloatTensor).cuda()
        x = F.relu(self.conv1(cwt))
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
model = ConvNet(Morlet_filter)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(112),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
def CNN_processing(seg_list):
    pred_list=[]
    for i, seg in enumerate(seg_list):
        # 創建 PILLOW 
        # image = Image.fromarray((image * 255).astype(np.uint8))
        # input_data = transform(image).unsqueeze(0).to(device)
        seg = seg[None,None,:]
        num_examples  = seg.shape[0]
        signal_length = seg.shape[-1]
        seg = torch.from_numpy(seg).type(torch.FloatTensor)
        seg.requires_grad_(requires_grad=False)
        # 進行推理
        with torch.no_grad():
            seg=seg.cuda()
            output = model(seg)
            pred = output.argmax(dim=1)

        # 列印結果
        a=np.array(['F', 'N', 'Q','V'])
        i=pred.item()
        pred_list.append(a[i])
    return pred_list