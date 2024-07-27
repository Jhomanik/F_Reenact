"""
"""

import os
import json
import torch
import time
import numpy as np
import pdb
import cv2
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.utilities.utils import *
from libs.utilities.image_utils import *
from libs.DECA.estimate_DECA import DECA_model
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import calculate_evaluation_metrics
from libs.utilities.dataloader import CustomDataset_validation
from libs.face_models.landmarks_estimation import LandmarksEstimation
from libs.models.hypermodel import HyperModel
from libs.models.stylemodel import StyleModel
from torchmetrics.image.fid import FrechetInceptionDistance

class Evaluator(object):
    def __init__(self, args):    
        
        self.exp_ranges = np.load('./libs/configs/ranges_FFHQ.npy')
        self.initialize_arguments(args)

    def initialize_arguments(self, args):
        self.model_path = args['model_path']
        self.model_type = args['model_type']
        self.image_resolution = args['image_resolution']
        self.dataset_path = args['dataset_path']
        self.test_batch_size = args['test_batch_size']
        self.workers = args['workers']
        self.eval_type = args['eval_type']

    def load_models(self):
    ################## Initialize models #################
        print('-- Load DECA model ')
        self.deca = DECA_model('cuda')
        self.id_loss_ = id_loss.IDLoss().cuda().eval()
        self.lpips_loss_ = LPIPS(net_type='alex').cuda().eval()
        self.landmarks_est = LandmarksEstimation(type='2D', path_to_detector='./pretrained_models/s3fd-619a316812.pth')
        if self.model_type == 'Hyper':
            self.model = HyperModel(self.model_path)
        elif self.model_type == 'Style':
            self.model = StyleModel(self.model_path)
        self.model.load_models()



    def configure_dataset(self):
        self.test_dataset = CustomDataset_validation(dataset_path = self.dataset_path, eval_type = self.eval_type)	
        self.test_dataloader = DataLoader(self.test_dataset,
									batch_size=self.test_batch_size ,
									shuffle=False,
									drop_last=True)
    def load_image(self, image):	
        image = preprocess_image(image, self.landmarks_est)
        image = image_to_tensor(image).unsqueeze(0).cuda()
        return image

    def evaluate_model_reenactment(self):
        self.model.eval()
        exp_error = 0; pose_error = 0; csim_total = 0; lpips_total = 0; count = 0
        fid = FrechetInceptionDistance(feature=64).to('cuda')
        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            with torch.no_grad():
                imgs_source, imgs_target = batch
                imgs_shifted = self.model(imgs_source, imgs_target)
                params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
                params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
                params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
                csim, pose, exp, lpips = calculate_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source, self.id_loss_, self.lpips_loss_, self.exp_ranges)
                fid.update(imgs_target.to(torch.uint8).to('cuda'), real=True)
                fid.update(imgs_shifted.to(torch.uint8).to('cuda'), real=False)
                exp_error += exp
                csim_total += csim
                pose_error += pose
                lpips_total += lpips
                
                count += 1

            if count % 100 == 0:
                print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}\t LPIPS: {:.2f} \t FID: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count, lpips_total/count, fid.compute()))

				

        print('*************** Validation ***************')
        print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}\t LPIPS: {:.2f} \t FID: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count, lpips_total/count, fid.compute()))
        print('*************** Validation ***************')

	
