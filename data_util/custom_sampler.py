import numpy as np

from torch.utils.data import Sampler

class VideoFrameSampler(Sampler):
    
    def __init__(self, video_indices, passes_per_epoch, max_frames_per_video):
        self.video_indices = video_indices
        self.video_iterator = (i for i in np.random.permutation(video_indices))
        self.length = len(list(self.video_iterator))*passes_per_epoch
        self.max_frames_per_video = max_frames_per_video
        
    def __iter__(self):
        for _ in range(len(self)):
            try:
                i = next(self.video_iterator)
            except StopIteration:
                self.video_iterator = (i for i in np.random.permutation(self.video_indices))
                i = next(self.video_iterator)
            j = np.random.randint(0, self.max_frames_per_video)
            yield (i, j)
        
    def __len__(self):
        return self.length