from torchvision import transforms
from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
img_path="hymenoptera_data/train/ants/7759525_1363d24e88.jpg"
img=Image.open(img_path)

#to_tensor
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
print(tensor_img)

Writer=SummaryWriter("logs")
Writer.add_image("tensor_img",tensor_img)
Writer.close()

#Normalize
print(tensor_img[0][0][0])
trans_nol=transforms.Normalize([1,2,3],[1,1,1])
img_nol=trans_nol(tensor_img)
print(img_nol[0][0][0])

ws=SummaryWriter("ll1")
ws.add_image("normalize",img_nol,1)
ws.close()

#Resize -Compose
trans_resize=transforms.Resize(512)
trans_cmp=transforms.Compose([trans_resize,tensor_trans])
resize_img=trans_cmp(img)
print(resize_img.shape)
ws.add_image("resize",resize_img,2)
ws.close()