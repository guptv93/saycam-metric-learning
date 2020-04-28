import torch
from torch import nn
import torch.nn.functional as F

conv1_in_ch = 1
conv2_in_ch = 20
fc1_in_features = 50*4*4
fc2_in_features = 500
h_dim = 10

class MNISTNet(nn.Module):
    def __init__(self, z_dim):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_in_ch, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=conv2_in_ch, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=500)
        self.fc2 = nn.Linear(in_features=fc2_in_features, out_features=h_dim)
        self.g = nn.Linear(in_features=h_dim, out_features=z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, fc1_in_features) # reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.g(x)
        return x
    
conv1_in_ch_mov = 3
conv2_in_ch_mov = 20
fc1_in_features_mov = 50*4*4
fc2_in_features_mov = 500
h_dim_mov = 30

class MovMNISTNet(nn.Module):
    def __init__(self, z_dim):
        super(MovMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_in_ch_mov, out_channels=conv2_in_ch_mov, kernel_size=5, stride=2) #20x40x40
        self.conv2 = nn.Conv2d(in_channels=conv2_in_ch_mov, out_channels=50, kernel_size=5, stride=2) #50x8x8
        self.fc1 = nn.Linear(in_features=fc1_in_features_mov, out_features=fc2_in_features_mov)
        self.fc2 = nn.Linear(in_features=fc2_in_features_mov, out_features=h_dim_mov)
        self.g = nn.Sequential(nn.Linear(h_dim_mov, 10), nn.ReLU(), nn.Linear(10, z_dim))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, fc1_in_features) # reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.g(x)
        return x