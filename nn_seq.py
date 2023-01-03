import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class JX(nn.Module):
    def __init__(self):
        super(JX, self).__init__()
        self.model=Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,input):
        output=self.model(input)
        return output

jx=JX()
input=torch.ones(64,3,32,32)
output=jx(torch.ones(64,3,32,32))
print(output.shape)

writer=SummaryWriter("log_seq")
writer.add_graph(jx,input)
writer.close()