#!/usr/bin/env python
# coding: utf-8

# # Metric Learning on Video Frames using N-Pair Loss

# ### Import Libraries and Check GPU

# In[ ]:


import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from IPython.display import clear_output
from time import sleep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ### Directory Setup for PyTorch DatasetFolder

# In[ ]:


directory = ['/scratch/vvg239/headcam/frames', '/scratch/vvg239/headcam/moving_mnist/mm_folder']

def folder_setup(dir_name):
    """One Time Directory Setup for DatasetFolder"""
    file_list = os.listdir(dir_name)
    for file in file_list:
        fdr = dir_name + file[6:-4]
        os.mkdir(fdr, mode = 0o700)
        os.rename(dir_name + file, fdr + "/" + file)
        
#folder_setup(directory[1])


# ### Moving MNIST / Saycam Visualization

# In[ ]:


use_moving_mnist = True #6,8,2,9,1,8
filepath = ['frames/0/video_0.npy', 'moving_mnist/mm_folder/0/np_0.npy']
frames = np.load(filepath[use_moving_mnist])
print(frames.shape)

def plot_videoframes(frames):
    for i in range(min(frames.shape[0], 40)):
        plt.imshow(frames[i, :, :, :], interpolation='nearest')
        plt.show()
        sleep(0.01)
        clear_output(wait=True)

    plt.close()

plot_videoframes(frames)


# ### Setting up PyTorch Dataset

# In[ ]:


class NpyVideoFolder(datasets.DatasetFolder):
    
    def __init__(self, root, transform=None, distance=5, max_distance=15, update_after=0):
        super().__init__(root, loader=self.default_loader, transform=None, extensions='.npy')
        self.counter = 0
        self.distance = distance
        self.max_distance = max_distance
        self.update_after = update_after
        self.preprocess = transform
    
    def default_loader(self, path):
        """Load Numpy Files into PyTorch Dataset. Returns array of the form 2x3xWxH (as used in PyTorch)"""
        self.counter += 1
        if(self.counter == self.update_after):
            self.counter = 0
            self.distance += 1
            self.distance = min(self.distance, self.max_distance)
            print("Distance is %d!!" % self.distance)
        samples = torch.from_numpy(np.load(path))
        x = np.random.randint(samples.shape[0]-self.distance)
        sample_a = self.preprocess(samples[x].permute(2,0,1))
        sample_p = self.preprocess(samples[x + self.distance].permute(2,0,1))
        return torch.stack([sample_a,sample_p],dim=0)
    
transform = transforms.Compose([transforms.ToPILImage(), 
                                transforms.Resize(224), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

dataset = NpyVideoFolder(root=directory[use_moving_mnist], transform=transform, distance=15, max_distance=20, update_after=0)


# ### Setting up PyTorch Data Loader for Training and Validation

# In[ ]:


validation_split = 0.2
batch_size = 30
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

## Frame Visualization Helper
def viz_helper(data_loader, num_vid):
    batch, _ = iter(data_loader).next()
    video_frames = batch[num_vid]
    video_frames = video_frames.permute(0,2,3,1)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(video_frames[0])
    ax2.imshow(video_frames[1])

viz_helper(val_loader, 2)


# ### Setting the Network and N-Pair Loss Function

# In[ ]:


model = models.alexnet()
classifier = nn.Sequential(#nn.Dropout(0.1),
                           nn.Linear(9216,4096),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.1),
                           nn.Linear(4096,20))
model.classifier = classifier
#print(torch.mean(model.features[8].weight))

# model = models.resnet50()
# model.fc = nn.Linear(2048,100)
# print(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


def minibatch_n_pair_loss(batch):
    loss = 0.0
    N = batch.shape[0]//2
    for i in range(N):
        negative_samples = torch.cat((batch[N:N+i], batch[N+i+1:]),dim=0)
        loss += n_pair_loss(batch[i], batch[N+i], negative_samples) #+ n_pair_loss(batch[N+i], batch[i], negative_samples)
    return loss/N

def n_pair_loss(anchor, positive, negatives):
    #sim_pos = F.cosine_similarity(positive.unsqueeze(0), anchor.unsqueeze(0), dim=1)
    #sim_negs = F.cosine_similarity(negatives, anchor.repeat(negatives.shape[0],1), dim=1)
    #logits = torch.cat((sim_pos,sim_negs)).unsqueeze(0)
    n_pair = torch.cat((positive.unsqueeze(0), negatives))
    sims = F.cosine_similarity(n_pair, anchor.repeat(n_pair.shape[0], 1), dim=1)
    logits = sims.unsqueeze(0)
    labels = torch.zeros(1).to(device, dtype=torch.long)
    ret = F.cross_entropy(logits, labels)
    return ret

def validate(batch):
    N = batch.shape[0]//2
    anchors = batch[0:N]
    others = batch[N:]
    count = 0
    for i in range(N):
        sims = F.cosine_similarity(others, anchors[i].repeat(N, 1), dim=1)
        if sims.argmax() == i:
            count += 1
    return count


# ### Training the Network

# In[ ]:


def train(epoch):
    model.train()
    running_loss = 0
    lr = optimizer.param_groups[0]['lr']
    for batch_frames, _ in train_loader:
        optimizer.zero_grad()
        x = batch_frames[:,0]
        y = batch_frames[:,1]
        minibatch = torch.cat((x,y),dim=0)
        minibatch = minibatch.to(device)
        minibatch = model(minibatch)
        loss = minibatch_n_pair_loss(minibatch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch: %3d | Running Loss: %f | LR : %.9f" 
          % (epoch, running_loss, lr))


def validation():
    scheduler.step()
    accuracy = 0
    for batch_frames, _ in val_loader:
        with torch.no_grad():
            x = batch_frames[:,0]
            y = batch_frames[:,1]
            minibatch = torch.cat((x,y),dim=0)
            minibatch = minibatch.to(device)
            accuracy += validate(model(minibatch))
    return accuracy


# In[ ]:


epochs = 400
max_accuracy = 0
for epoch in range(epochs):
    train(epoch)
    accuracy = validation()
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model_file = 'model.pth'
        print("Saving model to %s. Accuracy = %d/%d." %(model_file, accuracy, split))
        torch.save(model.state_dict(), model_file)


# ### Visualize Network Performance

# In[ ]:


state_dict = torch.load(model_file)
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    batch_frames, _ = next(iter(val_loader))
    x = batch_frames[:,0]
    y = batch_frames[:,1]
    minibatch = torch.cat((x,y),dim=0)
    minibatch_gpu = minibatch.to(device)
    batch = model(minibatch_gpu)
    N = batch.shape[0]//2
    anchors = batch[0:N]
    others = batch[N:]
    minibatch = minibatch.permute(0,2,3,1)
    for i in range(N):
        sims = F.cosine_similarity(others, anchors[i].repeat(N, 1), dim=1)
        _, indices = torch.topk(sims, 3)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
        ax1.imshow(minibatch[i])
        ax2.imshow(minibatch[N+indices[0]])
        ax3.imshow(minibatch[N+indices[1]])
        ax4.imshow(minibatch[N+indices[2]])


# ### Trial Code

# In[ ]:


### Loss Function Sanity
# test = torch.zeros(6,3)
# test[0,0] = 1.0
# test[1,1] = 1.0
# test[2,2] = 1.0
# test[3,1] = 1.0
# test[4,0] = 1.0
# test[5,2] = 1.0
# print(validate(test))


# In[ ]:


### Single Training Step
# model.train()
# batch_frames, _ = iter(data_loader).next()
# optimizer.zero_grad()
# x = batch_frames[:,0]
# y = batch_frames[:,1]
# minibatch = torch.cat((x,y),dim=0)
# minibatch = minibatch.to(device)
# minibatch = model(minibatch)
# loss = minibatch_n_pair_loss(minibatch)
# loss.backward()
# optimizer.step()


# In[ ]:


# x = torch.arange(10., 20.)
# values, indices = torch.topk(x, 3)
# print(str(indices))

