import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ResidualBlock

class ResNet(nn.Module):
  def __init__(self, ch, num_classes=10):
      super(ResNet, self).__init__()
      self.in_channels = 16
      self.conv = nn.Conv2d(ch, 16, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn = nn.BatchNorm2d(16)
      self.layer1 = self.make_layer(16, 3, stride=2) # 1
      self.layer2 = self.make_layer(32, 3, stride=2)
      self.layer3 = self.make_layer(64, 3, stride=2)
      self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(64, num_classes)

  def make_layer(self, out_channels, num_blocks, stride):
      layers = []
      layers.append(ResidualBlock(self.in_channels, out_channels, stride))
      self.in_channels = out_channels
      for _ in range(num_blocks - 1):
          layers.append(ResidualBlock(out_channels, out_channels))
      return nn.Sequential(*layers)

  def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
  


class MLPModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    


class ResNetMLPModel(nn.Module):
    def __init__(self, ch=14, num_classes=10):
        super(ResNetMLPModel, self).__init__()
        self.resnet_model = ResNet(ch, num_classes)
        self.mlp_model = MLPModel(num_classes)
        self.fc = nn.Linear(num_classes*2, num_classes)  # 將兩個特徵向量合併進行分類
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, image, numbers):

        resnet_features = self.resnet_model(image)
        mlp_features = self.mlp_model(numbers)
        
        combined_features = torch.cat((resnet_features, mlp_features), dim=1)
        output = self.fc(combined_features)
        return self.Softmax(output)