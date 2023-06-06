import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Conv2d(nn.Module):
    """
    CONV, default activation is ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)):
        super(Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        output = output.mean(dim=0)
        output = self.fc(output)
        return output
    

