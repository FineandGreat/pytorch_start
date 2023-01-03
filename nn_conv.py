import torch
import torch.nn as nn
import torch.nn.functional as F
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
kenel=torch.tensor([[1,2,1],
                    [0,1,0],
                    [2,1,0]])
input=torch.reshape(input,(1,1,5,5))
kenel=torch.reshape(kenel,(1,1,3,3))

print(input.shape)
print(kenel.shape)

output=F.conv2d(input,kenel,stride=1,padding=1)
print(output)

output2=F.conv2d(input,kenel,stride=2,padding=1)
print(output2)