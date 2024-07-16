import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets,transforms,models

device = torch.device("cuda" if torch.cuda.is_available()else"cpu")

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#直接使用ImageNet的计算结果
    ]
)

train_dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)#训练开始前打乱数据

test_dataset = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32,shuffle=True)

model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)#使用目前准确度最高的预训练权重
model.fc = nn.Linear(model.fc.in_features,10)#修改全连接层，最后输出10种类别
model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters,lr=1e-3)
num_epoch = 10#训练轮次为10


