import torch
import torch.nn as nn
import torch.nn.functional as F


class Jx(nn.Module):
    def __init__(self):
        super(Jx, self).__init__()

    def forward(self,input):
        output=input*input
        return output

jx=Jx()
x=torch.tensor(11.0)
output=jx(x)
print(output)
