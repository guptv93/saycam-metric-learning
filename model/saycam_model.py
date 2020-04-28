import torch, torchvision
from torch import nn
import torch.nn.functional as F

class SimCLRMobileNet(nn.Module):
    def __init__(self, z_dim):
        super(SimCLRMobileNet, self).__init__()
        self.f = torchvision.models.mobilenet_v2(pretrained=False)
        self.f.classifier = nn.Identity()
        self.g = nn.Sequential(
            nn.Linear(1000, 500), 
            nn.ReLU(), 
            nn.Linear(500, z_dim)
        )

    def forward(self, x):
        h = self.f(x)
        z = self.g(F.relu(h))
        return z
