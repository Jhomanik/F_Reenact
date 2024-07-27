import os
import numpy as np
from PIL import Image
import torch
import warnings
import sys
from tqdm import tqdm
import argparse 
from argparse import Namespace
import random
import sys

from libs.face_models.landmarks_estimation import LandmarksEstimation
from libs.models.pose_encoder import DECAEncoder
from libs.models.appearance_encoder import ArcFaceEncoder
from libs.models.encoders.psp_encoders import Encoder4Editing
from libs.models.hypernetwork_reenact import Hypernetwork_reenact
from libs.DECA.decalib.datasets import datasets 


from libs.utilities.utils_inference import *
from libs.configs.config_models import *
from libs.utilities.utils import *


class HyperModel(nn.Module):

	def __init__(self, model_path):
		super(HyperModel, self).__init__()
		self.device = 'cuda'
		self.model_path = model_path
		
		####################################

		self.image_resolution = hyper_model_arguments['image_resolution']
		self.deca_layer = hyper_model_arguments['deca_layer']
		self.arcface_layer = hyper_model_arguments['arcface_layer']
		self.pose_encoder_path = hyper_model_arguments['pose_encoder_path']
		self.app_encoder_path = hyper_model_arguments['app_encoder_path']
		self.e4e_path = hyper_model_arguments['e4e_path']
		self.sfd_detector_path = hyper_model_arguments['sfd_detector_path']

	def load_auxiliary_models(self):
		self.landmarks_est =  LandmarksEstimation(type = '2D', path_to_detector = self.sfd_detector_path)
		
		################ Pose encoder ################
		print('********* Upload pose encoder *********')
		self.pose_encoder = DECAEncoder(layer = self.deca_layer).to(self.device) # resnet50 pretrained for DECA eval mode
		self.posedata = datasets.TestData()
		ckpt = torch.load(self.pose_encoder_path, map_location='cpu')
		d = ckpt['E_flame']					
		self.pose_encoder.load_state_dict(d)
		self.pose_encoder.eval()
		##############################################
		print('********* Upload appearance encoder *********')
		self.appearance_encoder = ArcFaceEncoder(num_layer = self.arcface_layer).to(self.device) # ArcFace model
		ckpt = torch.load(self.app_encoder_path, map_location='cpu')
		d_filt = {'facenet.{}'.format(k) : v for k, v in ckpt.items() }
		self.appearance_encoder.load_state_dict(d_filt)
		self.appearance_encoder.eval()
		#############################################

		print('********* Upload Encoder4Editing *********')
		self.encoder = Encoder4Editing(50, 'ir_se', self.image_resolution).to(self.device)
		ckpt = torch.load(self.e4e_path)
		self.encoder.load_state_dict(ckpt['e']) 
		self.encoder.eval()
	def load_models(self):

		self.load_auxiliary_models()

		print('********* Upload HyperReenact *********')
		opts = {}
		
		opts['device'] = self.device
		opts['deca_layer'] = self.deca_layer
		opts['arcface_layer'] = self.arcface_layer
		opts['checkpoint_path'] = self.model_path
		opts['output_size'] = self.image_resolution
		opts['channel_multiplier'] = hyper_model_arguments['channel_multiplier']
		opts['layers_to_tune'] = hyper_model_arguments['layers_to_tune']
		opts['mode'] = hyper_model_arguments['mode']
		opts['stylegan_weights'] = hyper_model_arguments['generator_weights']
		

		opts = Namespace(**opts)
		self.net = Hypernetwork_reenact(opts).to(self.device)
		self.net.eval()
		

		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()
	
		self.truncation = 0.7
		self.trunc = self.net.decoder.mean_latent(4096).detach().clone()

	def get_identity_embeddings(self, image):
		
		landmarks = get_landmarks(image, self.landmarks_est)
		id_hat, f_app = self.appearance_encoder(image, landmarks) # f_app 256 x 14 x 14 and id_hat 512
			
		return 	id_hat, f_app

	def get_pose_embeddings(self, image):
		# Preprocess like DECA the input image for pose encoder
		image_pose = image.clone()
		image_prepro = torch.zeros(image_pose.shape[0], 3, 224, 224).cuda()
		for k in range(image_pose.shape[0]):
			min_val = -1
			max_val = 1
			image_pose[k].clamp_(min=min_val, max=max_val)
			image_pose[k].add_(-min_val).div_(max_val - min_val + 1e-5)
			image_pose[k] = image_pose[k].mul(255.0).add(0.0) 
			image_prepro_, error_flag = self.posedata.get_image_tensor(image_pose[k])
			image_prepro[k] = image_prepro_		
		pose_hat, f_pose = self.pose_encoder(image_prepro) #  512, 28, 28

		return pose_hat, f_pose
	def forward(self, source_img, target_img):
		with torch.no_grad():	
			shifted_codes = self.encoder(source_img)				
			# Get identity embeddings
			id_hat, f_app = self.get_identity_embeddings(source_img)
			# Get pose embeddings
			pose_hat, f_pose = self.get_pose_embeddings(target_img)
			
			reenacted_image, shifted_codes, shifted_weights_deltas = self.net.forward(f_pose = f_pose,
														f_app = f_app, 
														codes = shifted_codes,
														truncation = self.truncation, trunc = self.trunc,
														return_latents=True,
														return_weight_deltas_and_codes=True
														)

		return reenacted_image
	
