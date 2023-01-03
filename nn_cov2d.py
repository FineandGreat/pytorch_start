import torch
import torch.nn as nn
import torchvision.datasets
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())

data_loader=DataLoader(data_set,batch_size=64)

class JX(nn.Module):
    def __init__(self):
        super(JX, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1)

    def forward(self,x):
        x=self.conv1(x)
        return x

jx=JX()
Writer=SummaryWriter("cov2d")
step=0
for data in data_loader:
    imgs,targets=data
    output=jx(imgs)
    #print(imgs.shape)
    #print(output.shape)
    output=torch.reshape(output,(-1,3,30,30))
    Writer.add_images("input",imgs,step)
    Writer.add_images("output",output,step)
    step=step+1
Writer.close()