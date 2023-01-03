import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

data_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(data_set,batch_size=64,drop_last=True)


class JX(nn.Module):
    def __init__(self):
        super(JX, self).__init__()
        self.l1=Linear(196608,10)
    def forward(self,input):
        output=self.l1(input)
        return output

jx=JX()
for data in data_loader:
    imgs,targets=data
    print(imgs.shape)
   # output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    print(output.shape)
    output=jx(output)
    print(output.shape)