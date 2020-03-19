import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


model = nn.Sequential(nn.Conv2d(3, 10, kernel_size=5),
                      nn.MaxPool2d(2, 2),
                      nn.ReLU(),
                      nn.Conv2d(10, 20, kernel_size=5),
                      nn.MaxPool2d(2,2),
                      nn.ReLU(),
                      Flatten(),
                      nn.Dropout(p=0.2),
                      nn.Linear(6480,200),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(200,50), #Represenation Layer
                      nn.ReLU(),
                      nn.Linear(50,10),
                      nn.ReLU(),
                      nn.Linear(10, 2)) #Contrastive Loss Layer