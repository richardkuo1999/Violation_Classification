import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
    
class ResNet(nn.Module):
  def __init__(self, ch, tokensize=10):
      super(ResNet, self).__init__()
      self.in_channels = 16
      self.conv = nn.Conv2d(ch, 16, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn = nn.BatchNorm2d(16)
      self.layer1 = self.make_layer(16, 3, stride=1)
      self.layer2 = self.make_layer(32, 3, stride=2)
      self.layer3 = self.make_layer(64, 3, stride=2)
      self.layer4 = self.make_layer(128, 3, stride=2)
      self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
      self.fc = nn.Linear(128*3*3, tokensize)

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
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
  

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out