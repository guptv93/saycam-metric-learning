# saycam-metric-learning
Applying Metric Learning (with Contrastive Loss) on Infant Vision Data

The mnist_model.py consists of our reduced neural model (consisting of 2 conv layers and 1 fully connected layer). The N Pair Ranking Loss has been implemented in metric_learning.ipynb. The checkpoint file can be loaded from this link:
https://drive.google.com/open?id=1JBvlHaZKen4ztgDZc-7dbdpxBEu9D1sP


The downstream task has been implemented in downstream.ipynb. Place the checkpoint file in a folder named `archive` created in the root directory. The downstream.ipynb will pick it up automatically.
