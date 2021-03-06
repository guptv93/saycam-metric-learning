import os
import os.path
import sys

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FramePairsDataset(Dataset):
    
    NUMPY_EXT = '.npy'
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    
    def __init__(self, root, extension, distance, transform, sort_key=None):
        self.root = root
        self.distance, self.transform = distance, transform
        self.classes = self._find_classes(self.root)
        self.sort_key = sort_key
        if extension == self.NUMPY_EXT:
            self.loader = self._npy_loader
        else:
            self.folders = self._folder_setup()
            self.loader = self._pil_loader
            
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        class_id, frame_id = index
        anchor, pos = self.loader(class_id, frame_id)
        if self.transform is not None:
            anchor, pos = self.transform(anchor), self.transform(pos)
        return anchor, pos, class_id
    
    def _npy_loader(self, class_id, frame_id):
        class_path = os.path.join(self.root, self.classes[class_id])
        #Select the .npy file for the video
        npy_file = [f.path for f in os.scandir(class_path) if f.is_file() and f.name.lower().endswith(self.NUMPY_EXT)][0]
        #Load the numpy video file
        frames = torch.from_numpy(np.load(npy_file))
        #Select the anchor and positive indices and frames
        fid, pid = self._get_id_pairs(frame_id, frames.shape[0])
        anchor, pos = frames[fid], frames[pid]
        #Change from HxWxC to CxHxW
        anchor, pos = anchor.permute(2,0,1), pos.permute(2,0,1)
        #Convert Image Tensor to PIL Image
        pil_converter = transforms.ToPILImage()
        return pil_converter(anchor), pil_converter(pos)
        
    def _pil_loader(self, class_id, frame_id):
        class_name = self.classes[class_id]
        image_list = self.folders[class_name]
        fid, pid = self._get_id_pairs(frame_id, len(image_list))
        anchor, pos = image_list[fid], image_list[pid]
        anchor, pos = Image.open(anchor), Image.open(pos)
        return anchor, pos
    
    def _folder_setup(self):
        images = {}
        dir = os.path.expanduser(self.root)
        for target in self.classes:
            d = os.path.join(dir, target)
            l = []
            for root, _, fnames in os.walk(d, followlinks=True):
                for fname in fnames:
                    if fname.lower().endswith(self.IMG_EXTENSIONS):
                        fname = os.path.join(root, fname)
                        l.append(fname)
            l = sorted(l, key=self.sort_key)
            images[target] = l
        return images
    
    def _get_id_pairs(self, frame_id, num_frames):
        if num_frames < 2:
            raise Exception('A class should have minimum two frames')
        frame_id = frame_id % num_frames
        if self.distance > 0:
            distance = np.random.randint(self.distance) + 1
        else:
            distance = 0 #static simclr setting
        if frame_id + distance < num_frames:
            return (frame_id, frame_id + distance)
        elif frame_id - distance > 0:
            return (frame_id, frame_id - distance)
        else:
            return (np.random.randint(0, num_frames//2), np.random.randint(num_frames//2, num_frames))        
    
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        return classes