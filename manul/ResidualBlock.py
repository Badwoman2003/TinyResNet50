import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,filters,stride = 1):
        super(ResidualBlock,self).__init__()
        F1,F2,F3 = filters
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=F1,kernel_size=1,stride=stride,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=F1,out_channels=F2,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels=F2,out_channels=F3,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(F3)
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=F3,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(F3),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        residual = self.res(x)
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output += residual
        output = self.relu(output)
        return output
    
class ResNet50(nn.Module):
    def __init__(self, ):
        super(ResNet50,self).__init__()
        

