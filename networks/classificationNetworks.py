### Torch related imports
import torch.nn as nn
from .resnet18_cifar10 import *


class CNNClassifierNetwork(nn.Module):
    def __init__(self):
        super(CNNClassifierNetwork, self).__init__()

        self.backbone = resnet18(pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x
