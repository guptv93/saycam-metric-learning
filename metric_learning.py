#!/usr/bin/env python
# coding: utf-8

# # Metric Learning on Video Frames using N-Pair Loss

# ### Import Libraries and Check GPU

# In[1]:


import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ### Folder Setup for PyTorch DatasetFolder

# In[2]:


filepath = ["moving_mnist/mm_folder/0/np_0.npy", "frames/0/video_0.npy"]

def folder_setup(dir_name):
    """One Time Directory Setup for DatasetFolder"""
    file_list = os.listdir(dir_name)
    for file in file_list:
        fdr = dir_name + file[6:-4]
        os.mkdir(fdr, mode = 0o700)
        os.rename(dir_name + file, fdr + "/" + file)
        
#folder_setup(filepath[1])


# ### Moving MNIST / Saycam Visualization

# In[3]:


import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

frames = torch.from_numpy(np.load(filepath[0]))
print(frames.shape)

def plot_movingmnist(frames):
    for i in range(min(frames.shape[0], 20)):
        plt.imshow(frames[i, :, :, :], interpolation='nearest')
        plt.show()
        sleep(0.01)
        clear_output(wait=True)

    plt.close()

plot_movingmnist(frames)


# ### Setting up PyTorch DataLoader

# In[4]:


directory = ['/scratch/vvg239/headcam/moving_mnist/mm_folder', '/scratch/vvg239/headcam/frames']

def npy_loader(path):
    """Load Numpy Files into PyTorch Dataset. Returns array of the form Nx3xWxH (as used in PyTorch)"""
    sample = torch.from_numpy(np.load(path))
    x = np.random.randint(sample.shape[0]-10)
    return sample[x:x+10].permute(0,3,1,2).float()    

def create_dataloader(num):
    dataset = datasets.DatasetFolder(
        root=directory[num],
        loader=npy_loader,
        #transform = transforms.Compose([]),
        extensions='.npy'
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=True, num_workers=5)
    return data_loader

## Frame Visualization Helper
def viz_helper(num, use_loader=True):
    data_loader = create_dataloader(num)
    if use_loader == True:
        batch, _ = iter(data_loader).next()
        video_frames = batch[0]
        video_frames = video_frames.permute(0,2,3,1)
    else:
        video_frames = np.load(filepath[num])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(video_frames[0])
    ax2.imshow(video_frames[8])
    
viz_helper(0, use_loader=True)
data_loader = create_dataloader(0)


# ### Setting the Network and N-Pair Loss Function

# In[5]:


model = models.alexnet()
classifier = nn.Sequential(#nn.Dropout(0.1),
                           nn.Linear(9216,4096),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.1),
                           nn.Linear(4096,10)
                          )
    
model.classifier = classifier
print(model)
#print(torch.mean(model.features[8].weight))
model.to(device)

def minibatch_n_pair_loss(batch):
    loss = 0.0
    N = batch.shape[0]//2
    for i in range(N):
        negative_samples = torch.cat((batch[N:N+i], batch[N+i+1:]),dim=0)
        loss += n_pair_loss(batch[i], batch[N+i], negative_samples) #+ n_pair_loss(batch[N+i], batch[i], negative_samples)
    return loss/N

def n_pair_loss(anchor, positive, negatives):
    sim_pos = F.cosine_similarity(positive.unsqueeze(0), anchor.unsqueeze(0), dim=1)
    sim_negs = F.cosine_similarity(negatives, anchor.repeat(negatives.shape[0],1), dim=1)
    #print("-ve sim dimensions: " + str(sim_negs.shape))
    #print("+ve sim dimensions:" + str(sim_pos.shape))
    logits = torch.cat((sim_pos,sim_negs)).unsqueeze(0)
    labels = torch.ones(1).to(device, dtype=torch.long)
    #print("logits:" + str(logits))
    ret = F.cross_entropy(logits, labels)
    return ret



optimizer = optim.Adam(model.parameters(), lr=0.00001)
train_losses = []
epochs = 10000
model.train()

for epoch in range(epochs):
    running_loss = 0
    for batch_frames, _ in data_loader:
        optimizer.zero_grad()
        x = batch_frames[:,0]
        y = batch_frames[:,8]
        minibatch = torch.cat((x,y),dim=0)
        minibatch = minibatch.to(device)
        minibatch = model(minibatch)
        loss = minibatch_n_pair_loss(minibatch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        train_losses.append(running_loss)
        print("Running Loss: " + str(running_loss))

print(train_losses)