import time
import torch
import torch.nn as nn
from fvcore.nn import parameter_count_table

import sys
sys.path.append('./')

from models.ResNet import ResNetMLPModel as Model

def build_model(ch=14, num_classes=2):
            
    model = Model(ch ,num_classes)
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
  model = build_model(30, 2)

  # loss function 
  criterion = nn.CrossEntropyLoss()

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


  image = torch.randn((15, 30, 640, 640))
  bbox = torch.randn((15, 4), dtype=torch.float32)

  # 設定目標標籤
  target = torch.tensor([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],dtype=torch.float32)

  t1 = time.time()
  output = model(image, bbox)
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