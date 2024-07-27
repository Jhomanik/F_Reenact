import os
import datetime
import random
import sys
import argparse
from argparse import Namespace
import torch
from torch import nn
import numpy as np
import warnings
from tqdm import tqdm
import face_alignment

from libs.models.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.utils import make_noise, generate_image, generate_new_stylespace
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import stylegan2_ffhq_1024
from libs.utilities.utils_inference import invert_image
from libs.utilities.utils import preprocess_image
from libs.utilities.image_utils import image_to_tensor
from libs.models.encoders.psp import pSp

class StyleModel(nn.Module):

	def __init__(self, masknet_path):
		super(StyleModel, self).__init__()
		self.device = 'cuda'
		self.masknet_path = masknet_path
		self.image_resolution = 1024
		self.resize_image = True


	def load_models(self, inversion = True):

		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		
		self.generator_path = stylegan2_ffhq_1024['gan_weights'] 
		self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
		self.split_sections = stylegan2_ffhq_1024['split_sections']
		self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']
		
		if os.path.exists(self.generator_path):
			print('----- Load generator from {} -----'.format(self.generator_path))
					
			self.G = StyleGAN2Generator(256, 512, 8, channel_multiplier = 1)
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = False)
			self.G.cuda().eval()
			# use truncation 
			self.truncation = 0.7     
			self.trunc =self.G.mean_latent(4096).detach().clone()			
			
		else:
			print('Please download the pretrained model for StyleGAN2 generator and save it into ./pretrained_models path')
			exit()

		if os.path.exists(self.masknet_path):
			print('----- Load mask network from {} -----'.format(self.masknet_path))
			ckpt = torch.load(self.masknet_path, map_location=torch.device('cpu'))
			self.num_layers_control = ckpt['num_layers_control']
			self.mask_net = nn.ModuleDict({})
			for layer_idx in range(self.num_layers_control):
				network_name_str = 'network_{:02d}'.format(layer_idx)

				# Net info
				stylespace_dim_layer = self.split_sections[layer_idx]	
				input_dim = stylespace_dim_layer
				output_dim = stylespace_dim_layer
				inner_dim = stylespace_dim_layer

				network_module = MaskPredictor(input_dim, output_dim, inner_dim = inner_dim)
				self.mask_net.update({network_name_str: network_module})
			self.mask_net.load_state_dict(ckpt['mask_net'])
			self.mask_net.cuda().eval()
		else:
			print('Please download the pretrained model for Mask network and save it into ./pretrained_models path')
			exit()
		
		if inversion:
			### Load inversion model only when the input is image. ###
			self.encoder_path = stylegan2_ffhq_1024['e4e_inversion_model']
			print('----- Load e4e encoder from {} -----'.format(self.encoder_path))
			ckpt = torch.load(self.encoder_path, map_location='cpu')
			opts = ckpt['opts']
			opts['output_size'] = self.image_resolution
			opts['checkpoint_path'] = self.encoder_path
			opts['device'] = 'cuda'
			opts['channel_multiplier'] = self.channel_multiplier
			
			opts = Namespace(**opts)
			self.encoder = pSp(opts)
			self.encoder.cuda().eval()


	def reenact_pair(self, source_code, target_code):
		
		with torch.no_grad():
			# Get source style space
			source_img, style_source, w_source, noise_source = generate_image(self.G, source_code, self.truncation, self.trunc, 256, self.split_sections,
					input_is_latent = True, return_latents= True, resize_image = self.resize_image)

			# Get target style space
			target_img, style_target, w_target, noise_target = generate_image(self.G, target_code, self.truncation, self.trunc, 256, self.split_sections,
					input_is_latent = True, return_latents= True, resize_image = self.resize_image)
		
			# Get reenacted image
			
			masks_per_layer = []
			for layer_idx in range(self.num_layers_control):
				network_name_str = 'network_{:02d}'.format(layer_idx)
				style_source_idx = style_source[layer_idx]
				style_target_idx = style_target[layer_idx]			
				styles = style_source_idx - style_target_idx
				mask_idx = self.mask_net[network_name_str](styles)
				masks_per_layer.append(mask_idx)

			mask = torch.cat(masks_per_layer, dim=1)
			style_source = torch.cat(style_source, dim=1)
			style_target = torch.cat(style_target, dim=1)

			new_style_space = generate_new_stylespace(style_source, style_target, mask, num_layers_control = self.num_layers_control)
			new_style_space = list(torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
			reenacted_img = decoder(self.G, new_style_space, w_source, noise_source, resize_image = self.resize_image)
			
		return source_img, target_img, reenacted_img

	def forward(self, source_img, target_img):	
		inv_image, source_code = invert_image(source_img, self.encoder, self.G, self.truncation, self.trunc)
		
		inv_image, target_code = invert_image(target_img, self.encoder, self.G, self.truncation, self.trunc)
		source_img, target_img, reenacted_img = self.reenact_pair(source_code, target_code)
		return reenacted_img

	
