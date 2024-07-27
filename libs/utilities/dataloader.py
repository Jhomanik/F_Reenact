"""
"""
import torch
import os
import glob
import cv2
import random
import numpy as np
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import Dataset
from libs.utilities.image_utils import *
from libs.utilities.utils import *
from random import shuffle
from libs.face_models.landmarks_estimation import LandmarksEstimation
class CustomDataset_validation(Dataset):
    def __init__(self, dataset_path, eval_type = 'rec', size = None, shuffle = False, cherry = False):
        self.landmarks_est = LandmarksEstimation(type='2D', path_to_detector='./pretrained_models/s3fd-619a316812.pth')
        self.dataset_path = dataset_path
        if eval_type == 'rec' or eval_type == 'cross':
            self.frames = []
            for f in os.listdir(self.dataset_path):
    
                filename, file_extension = os.path.splitext(f)
                if file_extension == '.jpg' or file_extension == '.png':
                    self.frames.append(os.path.join(self.dataset_path, f))
            
    
          
            self.target_frames = self.frames[:]
          
            if eval_type == 'cross':
                if cherry:
                    random.Random(228).shuffle(self.target_frames)
                else:
                    random.shuffle(self.target_frames)
        

    
        if eval_type == 'self':
            video_dict = {}
            for f in os.listdir(self.dataset_path):
                filename, file_extension = os.path.splitext(f)
                if file_extension != '.jpg' and file_extension != '.png':
                    continue
                num = int(f.split("_")[0])
                frame = os.path.join(self.dataset_path, f)
                if num not in video_dict.keys():
                    video_dict[num] = [frame]
                else:
                    video_dict[num].append(frame) 
            self.frames = []
            self.target_frames = []
            for key in video_dict.keys():
                num_videos = len(video_dict[key])
                self.target_frames.extend(video_dict[key])
                self.frames.extend([video_dict[key][0]]*num_videos)
                
        if shuffle:
            zipped = list(zip(self.frames, self.target_frames))
            random.shuffle(zipped)
            self.frames, self.target_frames = zip(*zipped)

        if size is None:
            self.size = len(self.frames)
        else:
            self.size = min(size, len(self.frames))
        
    def __len__(self):	
        return self.size
    def load_image(self, image):	
        image = preprocess_image(image, self.landmarks_est)
        image = image_to_tensor(image).cuda()
        return image
    def __getitem__(self, index):
        return self.load_image(self.frames[index % self.size]), self.load_image(self.target_frames[index % self.size])