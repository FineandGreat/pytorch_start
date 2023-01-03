import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

data_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader=DataLoader(data_set,batch_size=64)
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
loss=nn.CrossEntropyLoss()
jx=JX()
optim=torch.optim.SGD(jx.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0
    for data in data_loader:
        imgs,targets=data
        output=jx(imgs)
        result_loss=loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss
    print(running_loss)
