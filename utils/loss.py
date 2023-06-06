import torch
import torch.nn as nn


class Loss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, hyp):
        super().__init__()
        
        self.hyp = hyp
        self.nc = hyp['nc']
        weight = torch.tensor([2,1])
        # self.losses = nn.CrossEntropyLoss()
        self.losses = nn.CrossEntropyLoss(weight=weight)
        # self.losses = nn.BCEWithLogitsLoss()
        # self.losses = nn.BCEWithLogitsLoss(pos_weight=weight)
        

    def forward(self, output, target):
        return self.losses(output, target)