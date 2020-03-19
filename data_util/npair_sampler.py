import numpy as np

from torch.utils.data import Sampler

class VideoFrameSampler(Sampler):
    
    def __init__(self, num_videos, frames_per_epoch, max_frames_per_video):
        self.num_videos = num_videos
        self.frames_per_epoch = frames_per_epoch
        self.max_frames_per_video = max_frames_per_video
        self.video_iterator = (i for i in np.random.permutation(num_videos))
        #self.frame_iterators = [np.random.randint(0,max_frames_per_video) for _ in range(num_videos)]
        
    def __iter__(self):
        for _ in range(len(self)):
            try:
                i = next(self.video_iterator)
            except StopIteration:
                self.video_iterator = (i for i in np.random.permutation(self.num_videos))
                i = next(self.video_iterator)
            j = np.random.randint(0, self.max_frames_per_video)
            yield (i, j)
        
    def __len__(self):
        return self.num_videos*self.frames_per_epoch