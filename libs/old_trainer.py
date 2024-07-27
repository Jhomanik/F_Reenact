"""
"""

import os
import json
import torch
import time
import numpy as np
import pdb
import cv2
import wandb
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from libs.models.stylemodel import StyleModel
from torchmetrics.image.fid import FrechetInceptionDistance
from libs.utilities.utils import *
from libs.utilities.image_utils import *
from libs.utilities.utils_inference import calculate_evaluation_metrics
from libs.DECA.estimate_DECA import DECA_model
from libs.models.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.stylespace_utils import encoder, decoder
from libs.configs.config_models import *
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import generate_grid_image, calculate_evaluation_metrics
from libs.models.encoders.psp_encoders import Encoder4Editing, Encoder4StyleEditing
from libs.utilities.dataloader import CustomDataset_validation

class OldTrainer(object):

	def __init__(self, args):
		
		self.args = args
		self.initialize_arguments(args)
		################# Initialize output paths #################
		make_path(self.output_path)
		self.log_dir = os.path.join(self.output_path, 'logs')
		make_path(self.log_dir)	
		self.models_dir = os.path.join(self.output_path, 'models')
		make_path(self.models_dir)
		####################################################################
		
		# save arguments file with params
		#save_arguments_json(args, self.output_path, 'arguments.json')
	
	def initialize_arguments(self, args):

		self.output_path = args['experiment_path']
		self.use_wandb = args['use_wandb']
		self.log_images_wandb = args['log_images_wandb']
		self.project_wandb = args['project_wandb']

		self.image_resolution = args['image_resolution']
		self.type = args['type']
		self.train_dataset_path = args['train_dataset_path']
		self.test_dataset_path = args['test_dataset_path']

		self.epochs = args['epochs'] 
		self.lr = args['lr'] 
		self.num_layers_control = args['num_layers_control']
		self.max_iter = args['max_iter'] 
		self.batch_size = args['batch_size'] 
		self.test_batch_size = args['test_batch_size']


		# Weights
		self.lambda_identity = args['lambda_identity']
		self.lambda_perceptual = args['lambda_perceptual']
		self.lambda_shape = args['lambda_shape']
		self.lambda_pixel = args['lambda_pixel']
		
		self.steps_per_log = args['steps_per_log']
		self.steps_per_save_models = args['steps_per_save_models']
		self.steps_per_evaluation = args['steps_per_evaluation']
		self.steps_per_image_log = args['steps_per_image_log']

		self.validation_pairs = args['validation_pairs']
		# 	self.num_pairs_log = self.validation_pairs

		

	def load_models(self):

		################## Initialize models #################
		print('-- Load DECA model ')
		self.deca = DECA_model('cuda')
		self.id_loss_ = id_loss.IDLoss().cuda().eval()
		self.lpips_loss_ = LPIPS(net_type='alex').cuda().eval()
		self.losses = Losses()
		####################################################################
		self.e4e_path = hyper_model_arguments['source_e4e_path']
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		if self.image_resolution == 256:
			self.generator_path = hyper_model_arguments['generator_weights'] 
			self.channel_multiplier = hyper_model_arguments['channel_multiplier']
			self.split_sections = hyper_model_arguments['split_sections']
			#self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']
			self.exp_ranges = np.load(stylegan2_ffhq_1024['expression_ranges'])
			
		else:
			print('Incorect dataset type {} and image resolution {}'.format(self.type, self.image_resolution))

		if self.num_layers_control is not None:
			self.num_nets = self.num_layers_control
		else:
			self.num_nets = len(self.split_sections)
			

		print('-- Load generator from {} '.format(self.generator_path))
		self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier= self.channel_multiplier)
		if self.image_resolution == 256:
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = False)
		else:
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = True)
		self.G.cuda().eval()
		self.truncation = 0.7
		self.trunc = self.G.mean_latent(4096).detach().clone()


		print('-- Initialize mask predictor.')
		self.mask_net = nn.ModuleDict({})
		for layer_idx in range(self.num_nets):
			network_name_str = 'network_{:02d}'.format(layer_idx)
			# Net info
			stylespace_dim_layer = self.split_sections[layer_idx]
			
			input_dim = stylespace_dim_layer
			output_dim = stylespace_dim_layer
			inner_dim = stylespace_dim_layer
			network_module = MaskPredictor(input_dim, output_dim, inner_dim = inner_dim)
			self.mask_net.update({network_name_str: network_module})
			out_text = 'Network {}: ----> input_dim:{} - output_dim:{}'.format(layer_idx, input_dim, output_dim)
			print(out_text)
		print('-- Initialize encoders.')
		self.source_encoder = Encoder4Editing(50, 'ir_se', self.image_resolution).cuda()
		ckpt = torch.load(self.e4e_path)
		self.source_encoder.load_state_dict(ckpt['e'])
		self.target_encoder = Encoder4Editing(50, 'ir_se', self.image_resolution).cuda()
		ckpt = torch.load(self.e4e_path)
		self.target_encoder.load_state_dict(ckpt['e'])

	def configure_dataset(self):
		self.train_dataset = CustomDataset_validation(self.train_dataset_path, self.type)	
		
		self.train_dataloader = DataLoader(self.train_dataset,
									batch_size=self.batch_size,
									shuffle=True,
									drop_last=True)
		self.test_dataset = CustomDataset_validation(self.test_dataset_path, self.type, self.validation_pairs)	
		self.test_dataloader = DataLoader(self.test_dataset,
									batch_size=self.test_batch_size,
									shuffle=True,
									drop_last=True)


	def get_shifted_image(self, style_source, style_target, w, noise):
		# Generate shift
		masks_per_layer = []
		for layer_idx in range(self.num_nets):
			network_name_str = 'network_{:02d}'.format(layer_idx)
			style_source_idx = style_source[layer_idx]
			style_target_idx = style_target[layer_idx]				
			styles = style_source_idx - style_target_idx
			mask_idx = self.mask_net[network_name_str](styles)
			masks_per_layer.append(mask_idx)

		style_source = torch.cat(style_source, dim=1)
		style_target = torch.cat(style_target, dim=1)
		mask = torch.cat(masks_per_layer, dim=1)
		new_style_space = generate_new_stylespace(style_source, style_target, mask, self.num_layers_control)
		
		new_style_space = list(torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
		imgs_shifted = decoder(self.G, new_style_space, w, noise, resize_image = True)
		
		return imgs_shifted, new_style_space

	def train(self):
		
		self.load_models()
		if self.use_wandb:
			#########################
			config = self.args
			wandb.init(
				project= self.project_wandb,
				notes="",
				tags=["debug"],
				config=config,
			)
			name = self.output_path.split('/')[-1]
			wandb.run.name = name
			wandb.watch(self.mask_net, log="all", log_freq=500)
			#######################
		self.configure_dataset()
		#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.G.eval().cuda()
		self.mask_net.train().cuda() 
		self.source_encoder.eval().cuda() 
		self.target_encoder.eval().cuda() 
        
		optimizer = torch.optim.Adam([
			{'params': self.mask_net.parameters()}  
		], lr=self.lr, weight_decay=5e-4)
	
		self.truncation = 0.7
		latent_in = torch.randn(4096, 512).cuda()
		self.trunc = self.G.style(latent_in).mean(0, keepdim=True)



		list_loss = []
		step = 0
		for epoch in range(self.epochs):
			for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
				imgs_source, imgs_target = batch
				loss_dict = {}
				#self.G.zero_grad()
				optimizer.zero_grad()
				

				w_source = self.source_encoder(imgs_source)			
				w_target = self.target_encoder(imgs_target)
				
				input_is_latent = True
			
				######## Source images ########
				with torch.no_grad():
					noise_source = [None] * self.G.num_layers
					style_source, w_source, _ = encoder(self.G, w_source, self.truncation, self.trunc, size = self.image_resolution, input_is_latent = input_is_latent)
					params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
					#self.G.zero_grad()			
					######## Target images	 ########
					noise_target = [None] * self.G.num_layers
					style_target, w_target, _ = encoder(self.G, w_target, self.truncation, self.trunc, size = self.image_resolution, input_is_latent = input_is_latent)
					params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
				#self.G.zero_grad()				
				######## Generate reenacted image between source and target images ########
				imgs_shifted, new_style_space = self.get_shifted_image(style_source, style_target, w_source, noise_source)	
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
			
				
				loss, loss_dict = self.calculate_loss(params_source, params_shifted, params_target, imgs_source, imgs_shifted, imgs_target)

			

				############## Total loss ##############	
				list_loss.append(loss.data.item())
			
				loss.backward()
			
				optimizer.step()

			

				######### Evaluate #########
				if step % self.steps_per_log == 0:
					out_text = '[step {}]'.format(step)
					for key, value in loss_dict.items():
						out_text += (' | {}: {:.2f}'.format(key, value))
					out_text += '| Mean Loss {:.2f}'.format(np.mean(np.array(list_loss)))
					print(out_text)
				if step % self.steps_per_image_log == 0:
					grid = generate_grid_image(imgs_source, imgs_target, imgs_shifted)
					save_image(grid, os.path.join(self.log_dir, '{:06d}.png'.format(step)))
					if self.use_wandb:
						image_array = grid.detach().cpu().numpy()
						image_array = np.transpose(image_array, (1, 2, 0))
						images = wandb.Image(image_array)
						wandb.log({
							'step': step,
						})
						wandb.log({"train images": images})
			
				if step % self.steps_per_save_models == 0 and step > 0:
					self.save_model(step)
					
				if step % self.steps_per_evaluation == 0 and step > 0:
					self.evaluate_model_reenactment(step)
					

				if step % 500 == 0 and step > 0:
					list_loss = []

				if self.use_wandb:
					wandb.log({
						'step': step,
					})
					wandb.log(loss_dict)
	
				del imgs_source, imgs_shifted, imgs_target
				del params_source, params_shifted, params_target
				torch.cuda.empty_cache()
				step += 1
            
	def calculate_loss(self, params_source, params_shifted, params_target, imgs_source, imgs_shifted, imgs_target):
		loss_dict = {} 
		loss = 0
		
		############## Shape Loss ##############
		if self.lambda_shape !=0:

			coefficients_gt = {}	
			coefficients_gt['pose'] = params_target['pose']
			coefficients_gt['exp'] = params_target['alpha_exp']	
			coefficients_gt['cam'] = params_source['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_source['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)

			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted['pose']
			coefficients_reen['shape'] = params_shifted['alpha_shp']
			coefficients_reen['exp'] = params_shifted['alpha_exp']
			coefficients_reen['cam'] = params_shifted['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			
			loss_shape = self.lambda_shape *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = self.lambda_shape *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = self.lambda_shape * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			if self.type == 'rec' or self.type == 'self':
				loss_pixel = self.lambda_pixel * self.losses.calculate_pixel_wise_loss(imgs_target, imgs_shifted)
				loss_dict['loss_pixel'] = loss_pixel.data.item()
				loss += loss_pixel
			loss_dict['loss_shape'] = loss_shape.data.item()
			loss_dict['loss_eye'] = loss_eye.data.item()
			loss_dict['loss_mouth'] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye
		####################################################

		############## Identity losses ##############	
		if self.lambda_identity != 0:
			loss_identity = self.lambda_identity * self.id_loss_(imgs_shifted, imgs_source.detach())
			loss_dict['loss_identity'] = loss_identity.data.item()
			loss += loss_identity

		if self.lambda_perceptual != 0:
			imgs_target_255 = torch_range_1_to_255(imgs_target)
			imgs_shifted_255 = torch_range_1_to_255(imgs_shifted)
			loss_perceptual = self.lambda_perceptual * self.lpips_loss_(imgs_shifted_255, imgs_target_255.detach())
			loss_dict['loss_perceptual'] = loss_perceptual.data.item()
			loss += loss_perceptual

		loss_dict['loss'] = loss.data.item()
		return loss, loss_dict


	def save_model(self, step):
		state_dict = {
			'step': 				step,
			'mask_net': 			self.mask_net.state_dict(),
			'num_layers_control': 	self.num_layers_control
		}
		checkpoint_path = os.path.join(self.models_dir, 'mask_net_{:06d}.pt'.format(step))
		torch.save(state_dict, checkpoint_path)

	'Evaluate models for face reenactment and save reenactment figure'
	def evaluate_model_reenactment(self, step):

		input_is_latent = True
		self.mask_net.eval()

		exp_error = 0; pose_error = 0; csim_total = 0; lpips_total = 0; count = 0
		fid = FrechetInceptionDistance(feature=64).cuda()
		imgs_source_for_log = None; imgs_target_for_log = None; imgs_shifted_for_log = None;
		for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
			with torch.no_grad():
				imgs_source, imgs_target = batch
                
				w_source = self.source_encoder(imgs_source)		
				w_target = self.target_encoder(imgs_target)		
				input_is_latent = True

				noise_source = [None] * self.G.num_layers
				style_source, w_source, _ = encoder(self.G, w_source, self.truncation, self.trunc, size = self.image_resolution, input_is_latent = input_is_latent)
					
				noise_target = [None] * self.G.num_layers
                
				style_target, w_target, _ = encoder(self.G, w_target, self.truncation, self.trunc, size = self.image_resolution, input_is_latent = input_is_latent)
					
				imgs_shifted, new_style_space = self.get_shifted_image(style_source, style_target, w_source, noise_source)

				if imgs_source_for_log is None:
					imgs_source_for_log = imgs_source; imgs_target_for_log = imgs_target; imgs_shifted_for_log = imgs_shifted
				params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
				params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)

				csim, pose, exp, lpips = calculate_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source, imgs_target, self.id_loss_, self.lpips_loss_, self.exp_ranges)
				fid.update(imgs_target.to(torch.uint8).to('cuda'), real=True)
				fid.update(imgs_shifted.to(torch.uint8).to('cuda'), real=False)
				exp_error += exp
				csim_total += csim
				pose_error += pose
				lpips_total += lpips

				count += 1

		print('*************** Validation ***************')
		print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}\t LPIPS: {:.2f} \t FID: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count, lpips_total/count, fid.compute()))
		print('*************** Validation ***************')
		grid = generate_grid_image(imgs_source_for_log, imgs_target_for_log, imgs_shifted_for_log)
		save_image(grid, os.path.join(self.log_dir, 'eval_{:06d}.png'.format(step)))

		if self.use_wandb:
			wandb.log({
				'expression_error_eval': exp_error/count,
				'pose_error_eval': pose_error/count,
				'csim_eval': csim_total/count,
				'lpips_eval': lpips_total/count,
				'fid_eval': fid.compute(),
			})

			image_array = grid.detach().cpu().numpy()
			image_array = np.transpose(image_array, (1, 2, 0))
			images = wandb.Image(image_array)
			wandb.log({"eval images": images})

		self.mask_net.train()

