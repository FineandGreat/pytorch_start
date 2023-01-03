import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])
input=torch.reshape(input,(-1,1,2,2))

data_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(data_set,batch_size=64)
class JX(nn.Module):
    def __init__(self):
        super(JX, self).__init__()
        self.r1=ReLU()
        self.s1=Sigmoid()
    def forward(self,input):
        output=self.s1(input)
        return output

jx=JX()
Writer=SummaryWriter("sigmoid")
step=0
for data in data_loader:
    imgs,targets=data
    output=jx(imgs)
    print(output)
    Writer.add_images("test_sigmoid",output,step)
    step+=1
Writer.close()
