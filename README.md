## Metric Learning with Temporal Contrastive Loss on Video Datasets

### Overview

Metric learning is the task of learning a distance function over objects. The idea is to learn a function that maps input patterns into a target space such that the L1/L2 norm in the target space approximates the “semantic” distance in the input space. It is commonly used as a pretext task for Self Supervised Learning (with Image Classification being the downstream task). 

The loss functions used with metric learning are called Ranking Loss. Unlike other loss functions, which aim to learn to predict a label directly, the objective of Ranking Loss is to predict relative distances between inputs. 

In our current experiment we will focus on unlabeled video datasets. How do we define semantic similarity/dissimilarity on unlabeled dataset? We use the 'temporal smoothness' of natural videos for this. Temportal Smoothness refers to the inherent property of natural videos, by which frames separated by a small time window generally consist of the same objects (visual contents) with some spatial translations. We train our network to non-linearly map the raw images (video frames) to points in a low dimensional space so that the distance between these points is small if the raw images are close to each other (temporally) in the videos and large otherwise.

The idea of using temporal closeness for metric learning comes from the [this paper by Wang, Gupta](https://arxiv.org/abs/1505.00687). We have replaced the triplet loss used in this paper, with N Pair Loss. The project closely follows the setup mentioned in the [SimCLR paper](https://arxiv.org/abs/2002.05709).

### Setup

The project uses PyTorch deep learning framework. The project folder structure is as follows:

    .
    ├── data_util/    Contains custom DataSet class, Transform class and Sampler classes. 
    ├── loss/         Contains code for NPair Contrastive Loss.
    └── model/        Contains the deep learning model class.


`./data_util/custom_dataset.py` contains the custom DataSet class. It returns a tuple consisting of three tensors: (i) a minibatch of anchor frames (ii) corresponding positive frames (iii) class labels (video numberings).  It can work with the following data types:

1. PIL files : The data is organized in pytorch's ImageFolder format. Each video has its own folder and the frames are named by their video timestamps.
2. NPY files : Each video is converted into its own Numpy array of format `NxHxWxC`and placed in the root data folder. The frames are arranged timewise. 

You can divide your videos into multiple mini videos of the same time frame. A pair of frames from the same video will act as (anchor frame, positive frame) pair. The `len` method of this DataSet class returns the number of such mini videos in the dataset.

`./data_util/custom_sampler.py` contains the custom Sampler class. It is used to sample from a dataset of videos, so it uses a tuple of two integers for indexing. The first integer gives the video number and the second integer gives the frame number. 

`./test_upstream.ipynb` contains a toy example for learning embeddings for MNIST digits in 2D/3D metric space. Though the dataset used in this example is not a video dataset, you can think of it as consisting 10 different videos each made up of images of a particular digit in 0-9.

`./test_downstream.ipynb` trains a linear classifier on top of a frozen feature extractor trained with temporal contrastive loss (upstream task). 
