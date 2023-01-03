import torch
from torch.nn import L1Loss, MSELoss

input=torch.tensor([1,2,3],dtype=torch.float32)
output=torch.tensor([1,2,5],dtype=torch.float32)
input=torch.reshape(input,(1,1,1,3))
output=torch.reshape(output,(1,1,1,3))

loss=L1Loss(reduction='sum')
result=loss(input,output)

lossmse=MSELoss()
result_mes=lossmse(input,output)

print(result_mes)
print(result)
