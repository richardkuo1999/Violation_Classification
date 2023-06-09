import time
import torch
import torch.nn as nn
from collections import OrderedDict
from fvcore.nn import parameter_count_table

import sys
sys.path.append('./')

from models.ResNet import ResNet_18 as ResNet
from models.MLP import MLPModel


class ResNetMLPModel_Merge(nn.Module):
    def __init__(self, ch=[25,2], num_classes=2, tokensize=32,  
                        hidden_dim=64, num_layers=2, num_heads=4, dropout=0.2):
        super(ResNetMLPModel_Merge, self).__init__()
        
        self.img_model = ResNet(3, tokensize)
        self.lane_model = ResNet(ch[0], tokensize)
        self.drive_model = ResNet(ch[1], tokensize)
        self.mlp_model = MLPModel(tokensize)
        # self.fc = TransformerClassifier(tokensize, hidden_dim, num_classes, num_layers, num_heads, dropout)
        self.fc = nn.Linear(tokensize*4, num_classes)


    def forward(self, image, laneline, drivable, bbox):

        image_features = self.img_model(image)
        laneline_features = self.lane_model(laneline)
        drivable_features = self.drive_model(drivable)
        bbox_features = self.mlp_model(bbox)
        
        combined_features = torch.cat((image_features, laneline_features, 
                            drivable_features, bbox_features), dim=1)
        
        output = self.fc(combined_features)
        return output
    
class ResNetMLPModel_Split(nn.Module):
    def __init__(self, ch=[25,2], num_classes=2, tokensize=32):
        super(ResNetMLPModel_Split, self).__init__()
        ch = 3 + sum(ch)
        self.model = ResNet(ch, tokensize)
        self.mlp_model = MLPModel(tokensize)
        self.fc = nn.Linear(tokensize*2, num_classes)

    def forward(self, image, laneline, drivable, bbox):
        input = torch.cat((image, laneline,drivable), dim=1)
        image_features = self.model(input)
        bbox_features = self.mlp_model(bbox)
        
        combined_features = torch.cat((image_features, bbox_features), dim=1)
        
        output = self.fc(combined_features)

        return output

def build_model(ch=[2,25], num_classes=2, tokensize=32, split=False):
    if split:
        model = ResNetMLPModel_Split(ch ,num_classes, tokensize)
    else:
        model = ResNetMLPModel_Merge(ch ,num_classes, tokensize)

    print(parameter_count_table(model))
    return model

def get_optimizer(hyp, model):
    if hyp['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                momentum=hyp['momentum'], weight_decay=hyp['wd'],
                                nesterov=hyp['nesterov'])
    elif hyp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                                betas=(hyp['momentum'], 0.999))   
    return optimizer

    
if __name__ == '__main__':
    model = build_model([3,3], 2, 2, True).cuda()
    # print(model)

    # loss function 
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    image = torch.randn((15, 3, 256, 256)).cuda()
    laneline = torch.randn((15, 3, 256, 256)).cuda()
    drivable = torch.randn((15, 3, 256, 256)).cuda()
    bbox = torch.randn((15, 4), dtype=torch.float32).cuda()

    # 設定目標標籤
    target = torch.tensor([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],dtype=torch.float32).cuda()

    t1 = time.time()
    output = model(image, laneline, drivable, bbox)
    t2 = time.time()
    print(output)
    print(torch.argmax(output, dim=1))

    print(f'predict use time: {t2-t1}')


    # 計算損失
    loss = criterion(output, target)

    # 反向傳播和參數更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'loss use time:{time.time() - t2}')

    print(f'loss:{loss.item()}')