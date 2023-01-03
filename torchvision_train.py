import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_dataset=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

writer=SummaryWriter("torchvision_test")
for i in range(10):
    img,target=train_dataset[i]
    writer.add_image("test",img,i)
writer.close()