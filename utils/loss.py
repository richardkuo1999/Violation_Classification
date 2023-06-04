import torch
import torch.nn as nn


class Loss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, hyp, device):
        super().__init__()
        
        self.hyp = hyp
        self.nc = hyp['nc']
        weight = torch.tensor([1.5,0.5])
        # self.losses = nn.CrossEntropyLoss(weight=weight).to(device)
        # self.losses = nn.CrossEntropyLoss().to(device)
        self.losses = nn.BCEWithLogitsLoss()
        # self.losses = nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        

    def forward(self, output, target):
        return self.losses(output, target)