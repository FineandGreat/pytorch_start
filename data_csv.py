import os
import pandas as pd
import torch
os.makedirs(os.path.join('.','data'),exist_ok=True)
data_file=os.path.join('.','data','house_tint.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')    #列名
    f.write('NA,Pave,127500\n')
    f.write('2.0,NA,106000\n')
    f.write('4.0,NA,178100\n')
    f.write('NA,NA,140000\n')

data=pd.read_csv(data_file)
print(data)

#插值
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]#inputs取前两列全部
inputs=inputs.fillna(inputs.mean())#数值列中NAN以均值填充
print(inputs)

inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)

x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x)
print(y)