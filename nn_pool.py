import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))

class JX(nn.Module):
    def __init__(self):
        super(JX, self).__init__()
        self.nn_pool=MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,input):
        output=self.s1(input)
        return output

jx=JX()
Writer=SummaryWriter("sigmoid")
