### Torch related imports
import torch.nn as nn
import torch.nn.functional as F
from resnet18_cifar10 import resnet18


class CNNClassifierNetwork(nn.Module):
    def __init__(self):
        super(CNNClassifierNetwork, self).__init__()

        self.backbone = resnet18(pretrained=True)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)

    def forward(self, x):
        x = self.backbone(x)
        return x
