import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

model = nn.Sequential(nn.Conv2d(3, 10, kernel_size=5),
                      nn.MaxPool2d(4, 4),
                      nn.ReLU(True),
                      nn.Conv2d(10, 20, kernel_size=5),
                      nn.Dropout2d(p=0.1),
                      nn.MaxPool2d(3,3),
                      nn.ReLU(True),
                      Flatten(),
                      nn.Linear(5780,10)
                     )