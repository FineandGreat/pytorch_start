from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import  Image
img_path="hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)

print(type(img_array))
print(img_array.shape)
Writer=SummaryWriter("logs")
Writer.add_image("test",img_array,1,dataformats="HWC")
for i in range(100):
    Writer.add_scalar("y=x",i,i)
Writer.close()