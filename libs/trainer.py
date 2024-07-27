import os
import json
import torch
import time
import numpy as np
import pdb
import cv2
import pandas as pd
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
from libs.models.gan.StyleGAN2.model import Discriminator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.stylespace_utils import decoder_with_rgb, decoder_from_16x16
from libs.configs.config_models import *
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import generate_grid_image, calculate_evaluation_metrics
from libs.models.encoders.psp_encoders import Encoder4Editing, Encoder4StyleEditing
from libs.utilities.dataloader import CustomDataset_validation
from libs.optimizers import Ranger
from libs.models.fusers import ContentLayerDeepFast




class Trainer(object):

	def __init__(self, args):
		
		self.args = args
		self.initialize_arguments(args)
		################# Initialize output paths #################
		make_path(self.output_path)
		self.log_dir = os.path.join(self.output_path, 'logs')
		make_path(self.log_dir)	
		make_path(os.path.join(self.log_dir, 'train_images'))
		make_path(os.path.join(self.log_dir, 'eval_images'))
		make_path(os.path.join(self.log_dir, 'cherry_images'))
		make_path(os.path.join(self.log_dir, 'loss_and_metrics'))
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
		self.cherry_dataset_path = args['cherry_dataset_path']

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




	def load_models(self):

		################## Initialize models #################
		print('-- Load DECA model ')
		self.deca = DECA_model('cuda')
		self.id_loss_ = id_loss.IDLoss().cuda().eval()
		self.lpips_loss_ = LPIPS(net_type='alex').cuda().eval()
		self.losses = Losses()
		####################################################################
		self.source_e4e_path = hyper_model_arguments['source_e4e_path']
		self.target_e4e_path = hyper_model_arguments['target_e4e_path']
		self.mask_net_path = hyper_model_arguments['mask_net_path']
		self.fuser_path = hyper_model_arguments['fuser_path']
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		if self.image_resolution == 256:
			self.generator_path = hyper_model_arguments['generator_weights'] 
			self.channel_multiplier = hyper_model_arguments['channel_multiplier']
			self.split_sections = hyper_model_arguments['split_sections']
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
		self.mean_styles, self.mean_rgbs = self.G.mean_style(4096)
		self.mean_styles = [s.detach().clone() for s in self.mean_styles]
		self.mean_rbgs = [s.detach().clone() for s in self.mean_rgbs]
		self.default_noise = [getattr(self.G.noises, 'noise_{}'.format(i)) for i in range(self.G.num_layers)]
		self.truncation = 0.7
		self.trunc = self.G.mean_latent(4096).detach().clone()

		self.disc = Discriminator(size = 256, channel_multiplier= self.channel_multiplier).cuda()
		self.disc.load_state_dict(torch.load(self.generator_path)['d'], strict = False)





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
			if self.mask_net_path is not None:
				ckpt = torch.load(self.mask_net_path)
				self.mask_net.load_state_dict(ckpt['mask_net'], strict=False)

		print('-- Initialize encoders.')
		self.encoder = Encoder4StyleEditing(50, 'ir_se', self.image_resolution, 512, self.channel_multiplier).cuda()
		ckpt = torch.load(self.source_e4e_path)
		self.encoder.load_state_dict(ckpt['source_enc'], strict=False)



		print('-- Initialize fuser.')
		self.fuser_hidden_dim = 256
		self.fuser_diff = ContentLayerDeepFast(len=14, inp=128 + self.fuser_hidden_dim, out=self.fuser_hidden_dim)
		self.fuser_inner = ContentLayerDeepFast(len=14, inp=2*self.fuser_hidden_dim, out=self.fuser_hidden_dim)
		if self.fuser_path is not None:
			ckpt = torch.load(self.fuser_path)
			self.fuser_diff.load_state_dict(ckpt['fuser_diff'], strict=False)
			self.fuser_inner.load_state_dict(ckpt['fuser_inner'], strict=False)
			print('-- Load fuser.')



	def configure_dataset(self):
		self.train_dataset = CustomDataset_validation(self.train_dataset_path, self.type)	
		
		self.train_dataloader = DataLoader(self.train_dataset,
									batch_size=self.batch_size ,
									shuffle=True,
									drop_last=True)
		self.extra_dataset = CustomDataset_validation(self.train_dataset_path, 'self')	
		self.extra_dataloader = DataLoader(self.extra_dataset,
									batch_size=self.batch_size ,
									shuffle=True,
									drop_last=True)

		self.test_dataset = CustomDataset_validation(self.test_dataset_path, self.type, self.validation_pairs, True)	
		self.test_dataloader = DataLoader(self.test_dataset,
									batch_size=self.test_batch_size,
									shuffle=True,
									drop_last=True)


		if self.cherry_dataset_path is not None:
			self.cherry_dataset = CustomDataset_validation(self.cherry_dataset_path, self.type, shuffle=False, cherry = True)	
			self.cherry_dataloader = DataLoader(self.cherry_dataset,
									batch_size=len(self.cherry_dataset),
									shuffle=False,
									drop_last=True)

	def get_shifted_image(self, style_source, rgb_source, noise_source, style_target, return_with_16x16 = False):
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
		if self.type == 'rec':
			new_style_space = style_source
		else:
			new_style_space = generate_new_stylespace(style_source, style_target, mask, self.num_layers_control)
		
		new_style_space = list(torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
		imgs_shifted = decoder_with_rgb(self.G, new_style_space, rgb_source, noise_source, resize_image = False, return_with_16x16 = return_with_16x16)
		return imgs_shifted, new_style_space, rgb_source

	def generate_synthetic_batch(self, imgs_source, imgs_target):
		style_source, rgb_source = self.encoder(imgs_source, self.mean_styles, self.mean_rgbs)	

		style_target, rgb_target = self.encoder(imgs_target, self.mean_styles, self.mean_rgbs)

		syn_source = decoder_with_rgb(self.G, style_source, rgb_source, self.default_noise, resize_image = False, return_with_16x16 = False)

		syn_target = decoder_with_rgb(self.G, style_target, rgb_target, self.default_noise, resize_image = False, return_with_16x16 = False)
		syn_gt, style_syn, rgb_syn = self.get_shifted_image(style_source, rgb_source , self.default_noise, style_target)
		return syn_source, syn_target, syn_gt


	def calculate_16x16_diff(self, imgs_source, imgs_target):
			style_source, rgb_source = self.encoder(imgs_source, self.mean_styles, self.mean_rgbs)	
			
			style_target, rgb_target = self.encoder(imgs_target, self.mean_styles, self.mean_rgbs)

			_, source_16x16 = decoder_with_rgb(self.G, style_source, rgb_source, self.default_noise, resize_image = False, return_with_16x16 = True)
			
				
				
			[imgs_shifted, shifted_16x16], style_shifted, rgb_shifted = self.get_shifted_image(style_source, rgb_source , self.default_noise, style_target, return_with_16x16 = True)


			return  style_shifted, rgb_shifted, shifted_16x16 - source_16x16




	def source_to_target(self, imgs_source, imgs_target, with_stylemask = False):
		with torch.no_grad():
			if imgs_target is not  None:
				style_shifted, rgb_shifted, diff = self.calculate_16x16_diff(imgs_source, imgs_target)
			else:
				diff = torch.zeros((imgs_source.shape[0], self.fuser_hidden_dim, 64, 64)).cuda()
				
				style_shifted, rgb_shifted = self.encoder(imgs_source, self.mean_styles, self.mean_rgbs)	
			source_16x16_features = self.encoder.return_16x16_features(imgs_source)

		new_features_16x16 = self.fuser_diff(torch.cat((source_16x16_features, diff), 1))
		imgs_shifted = decoder_from_16x16(self.G, new_features_16x16, self.fuser_inner,  style_shifted, rgb_shifted, self.default_noise, resize_image = False)
		if with_stylemask:
			imgs_stylemask = decoder_with_rgb(self.G, style_shifted, rgb_shifted, self.default_noise, resize_image = False)
			return imgs_shifted, imgs_stylemask
		return imgs_shifted

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

		self.G.eval().cuda()
		self.mask_net.eval().cuda() 
		self.encoder.eval().cuda() 

		self.fuser_diff.train().cuda() 
		self.fuser_inner.train().cuda()
		self.disc.train().cuda()

		optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr  = self.lr)
		optimizer = torch.optim.Adam([
			{'params': self.fuser_diff.parameters()}, 
			{'params': self.fuser_inner.parameters()}, 
			#{'params': self.fuser_target.parameters()}, 
			#{'params': self.fuser_64x64.parameters()}, 
			#{'params': self.mask_net.parameters()}, 
            #{'params': self.source_encoder.parameters()},
            #{'params': self.target_encoder.parameters()},
		], lr=self.lr)
	
		



		list_loss = []
		df_history = []
		df_eval_history = []
		step = 0

		for epoch in range(self.epochs):
			extra_iterator = iter(self.extra_dataloader)

			for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
				imgs_source, imgs_target = batch
				#try:
					#imgs_source_extra, imgs_target_extra = next(extra_iterator)
				#except StopIteration:
					#extra_iterator = iter(self.extra_dataloader)
					#imgs_source_extra, imgs_target_extra = next(extra_iterator)



				loss_dict = {}

				optimizer.zero_grad()
				with torch.no_grad():
					params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
					params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
					syn_source, syn_target, gt = self.generate_synthetic_batch(imgs_source, imgs_target)

					#params_source_extra, angles_source_extra = calculate_shapemodel(self.deca, imgs_source_extra)
					#params_target_extra, angles_target_extra = calculate_shapemodel(self.deca, imgs_target_extra)




				syn = self.source_to_target(syn_source, syn_target)	
				params_syn, angles_syn = calculate_shapemodel(self.deca, syn)
				params_gt, angles_gt = calculate_shapemodel(self.deca, gt)

				loss_diff, loss_dict_diff = self.calculate_loss(params_gt, params_syn, params_gt, gt, syn, gt)


				imgs_rec = self.source_to_target(imgs_source, None)
				params_rec, angles_rec = calculate_shapemodel(self.deca, imgs_rec)
				loss_rec, loss_dict_rec = self.calculate_loss(params_source, params_rec, params_source, imgs_source, imgs_rec, imgs_source, name = "_rec")

				#imgs_extra = self.source_to_target(imgs_source_extra, imgs_target_extra)
				#params_extra, angles_extra = calculate_shapemodel(self.deca, imgs_extra)
				#loss_extra, loss_dict_extra = self.calculate_loss(params_source_extra, params_extra, params_target_extra, imgs_source_extra, imgs_extra, imgs_target_extra, name = "_self")


				criterion = nn.BCEWithLogitsLoss()
				#real_images = torch.cat((imgs_source, imgs_target), dim = 0).detach()
				real_images = imgs_source.detach()
				fake_images = self.source_to_target(imgs_source, imgs_target)	

				self.disc.eval()
				fake_logits = self.disc(fake_images)
				fake_labels = torch.ones(fake_images.size(0), 1).cuda()
				loss_GAN = criterion(fake_logits, fake_labels) 




				loss = loss_rec + loss_diff + loss_GAN
				loss_dict = dict(list(loss_dict_rec.items()) + list(loss_dict_diff.items())) # + list(loss_dict_extra.items()) 
				loss_dict['total_loss'] = loss.data.item()

				loss_dict['loss_gen_GAN'] = loss_GAN.data.item()


				############## Total loss ##############	
				list_loss.append(loss.data.item())
				loss.backward()
				optimizer.step()

				############## Disc loss ##############
				real_labels = torch.ones(real_images.size(0), 1).cuda()
				fake_labels = torch.zeros(fake_images.size(0), 1).cuda()
				self.disc.train()
				optimizer_disc.zero_grad()
				real_logits = self.disc(real_images)		
				fake_logits = self.disc(fake_images.detach())

				loss_disc = criterion(fake_logits, fake_labels) + criterion(real_logits, real_labels)
				loss_disc.backward()
				optimizer_disc.step()	
				loss_dict['loss_disc'] = loss_disc.data.item()




				######### Evaluate #########
				if step % self.steps_per_log == 0:
					out_text = '[step {}]'.format(step)
					for key, value in loss_dict.items():
						out_text += (' | {}: {:.2f}'.format(key, value))
					out_text += '| Mean Loss {:.2f}'.format(np.mean(np.array(list_loss)))
					df_history.append(loss_dict)
					print(out_text)

				if step % self.steps_per_image_log == 0:
					if self.cherry_dataset_path is not None:
						self.cherry_pick(step)
					grid = generate_grid_image(syn_source, syn_target, syn)
					save_image(grid, os.path.join(self.log_dir,'train_images' ,'{:06d}.png'.format(step)))
					if self.use_wandb:
						image_array = grid.detach().cpu().numpy()
						image_array = np.transpose(image_array, (1, 2, 0))
						images = wandb.Image(image_array)
						wandb.log({
							'step': step,
						})
						wandb.log({"train images": images})

					grid = generate_grid_image(imgs_source, imgs_source, imgs_rec)
					save_image(grid, os.path.join(self.log_dir,'train_images' ,'rec_{:06d}.png'.format(step)))
					if self.use_wandb:
						image_array = grid.detach().cpu().numpy()
						image_array = np.transpose(image_array, (1, 2, 0))
						images = wandb.Image(image_array)
						wandb.log({
							'step': step,
						})
						wandb.log({"train images rec": images})


				if step % self.steps_per_save_models == 0 and step > 0:
					self.save_model(step)
					pd.DataFrame(df_history).to_csv(os.path.join(self.log_dir,'loss_and_metrics' ,'loss_history.csv'), mode='w+')


				if step % self.steps_per_evaluation == 0 and step > 0:
					metrics = self.evaluate_model_reenactment(step)
					df_eval_history.append(metrics)
					pd.DataFrame(df_eval_history).to_csv(os.path.join(self.log_dir, 'loss_and_metrics', 'metrics_history.csv'), mode='w+')
					

				if step % 500 == 0 and step > 0:
					list_loss = []

				if self.use_wandb:
					wandb.log({
						'step': step,
					})
					wandb.log(loss_dict)

				del imgs_source, imgs_target, syn_source, syn_target, syn

				torch.cuda.empty_cache()
				step += 1



	def cherry_pick(self, step):
		for batch in self.cherry_dataloader:
			with torch.no_grad():
				imgs_source, imgs_target = batch
				imgs_shifted, imgs_stylemask = self.source_to_target(imgs_source, imgs_target, True)	
				grid = generate_grid_image(imgs_source, imgs_target, imgs_shifted, imgs_stylemask)
				save_image(grid, os.path.join(self.log_dir,'cherry_images','cherry_{:06d}.png'.format(step)))
				if self.use_wandb:
					image_array = grid.detach().cpu().numpy()
					image_array = np.transpose(image_array, (1, 2, 0))
					images = wandb.Image(image_array)
					wandb.log({"cherry images": images})

				imgs_rec = self.source_to_target(imgs_source, None)	
				grid = generate_grid_image(imgs_source, imgs_source, imgs_rec)
				save_image(grid, os.path.join(self.log_dir,'cherry_images','cherry_rec_{:06d}.png'.format(step)))
				if self.use_wandb:
					image_array = grid.detach().cpu().numpy()
					image_array = np.transpose(image_array, (1, 2, 0))
					images = wandb.Image(image_array)
					wandb.log({"cherry images rec": images})



	def calculate_loss(self, params_source, params_shifted, params_target, imgs_source, imgs_shifted, imgs_target,
                      lambda_shape = None, lambda_pixel = None, lambda_identity = None, lambda_perceptual = None, name = ""):
		if lambda_shape is None:
			lambda_shape = self.lambda_shape
		if lambda_pixel is None:
			lambda_pixel = self.lambda_pixel
		if lambda_identity is None:
			lambda_identity = self.lambda_identity
		if lambda_perceptual is None:
			lambda_perceptual = self.lambda_perceptual

		loss_dict = {} 
		loss = 0

		############## Shape Loss ##############
		if lambda_shape !=0:

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
			
			loss_shape = lambda_shape *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = lambda_shape *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = lambda_shape * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			loss_dict['loss_shape' + name] = loss_shape.data.item()
			loss_dict['loss_eye' + name] = loss_eye.data.item()
			loss_dict['loss_mouth' + name] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye
		
		if lambda_pixel != 0:
			loss_pixel = lambda_pixel * self.losses.calculate_pixel_wise_loss(imgs_target, imgs_shifted)
			loss_dict['loss_pixel' + name] = loss_pixel.data.item()
			loss += loss_pixel

		############## Identity losses ##############	
		if lambda_identity != 0:
			loss_identity = lambda_identity * self.id_loss_(imgs_shifted, imgs_source.detach())
			loss_dict['loss_identity' + name] = loss_identity.data.item()
			loss += loss_identity

		if lambda_perceptual != 0:
			imgs_target_255 = torch_range_1_to_255(imgs_target)
			imgs_shifted_255 = torch_range_1_to_255(imgs_shifted)
			loss_perceptual = lambda_perceptual * self.lpips_loss_(imgs_shifted_255, imgs_target_255.detach())
			loss_dict['loss_perceptual' + name] = loss_perceptual.data.item()
			loss += loss_perceptual

		loss_dict['loss' + name] = loss.data.item()
		return loss, loss_dict






	def save_model(self, step):
		state_dict = {
			'step': 				step,
			'fuser_diff': 			self.fuser_diff.state_dict(),
			'fuser_inner': 			self.fuser_inner.state_dict(),
			#'mask_net': 			self.mask_net.state_dict(),
            #'source_enc':          self.source_encoder.state_dict(),
            #'target_enc':          self.target_encoder.state_dict(),
			'num_layers_control': 	self.num_layers_control
		}
		checkpoint_path = os.path.join(self.models_dir, 'fuser_net.pt')
		torch.save(state_dict, checkpoint_path)





	'Evaluate models for face reenactment and save reenactment figure'
	def evaluate_model_reenactment(self, step):

		input_is_latent = True

		self.fuser_diff.eval()
		self.fuser_inner.eval()

		exp_error = 0; pose_error = 0; csim_total = 0; lpips_total = 0; count = 0
		fid = FrechetInceptionDistance(feature=64).cuda()
		imgs_source_for_log = None; imgs_target_for_log = None; imgs_shifted_for_log = None; imgs_stylemask_for_log = None
		for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
			with torch.no_grad():
				imgs_source, imgs_target = batch
                
				imgs_shifted, imgs_stylemask = self.source_to_target(imgs_source, imgs_target, True)	

				if imgs_source_for_log is None:
					imgs_source_for_log = imgs_source; imgs_target_for_log = imgs_target; imgs_shifted_for_log = imgs_shifted
					imgs_stylemask_for_log = imgs_stylemask
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
				del imgs_source, imgs_shifted, imgs_target
				del params_source, params_shifted, params_target
				torch.cuda.empty_cache()

		print('*************** Validation ***************')
		print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}\t LPIPS: {:.2f} \t FID: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count, lpips_total/count, fid.compute()))
		print('*************** Validation ***************')
		metrics = {
				'expression_error_eval': exp_error/count,
				'pose_error_eval': pose_error/count,
				'csim_eval': csim_total/count,
				'lpips_eval': lpips_total/count,
				'fid_eval': fid.compute().data.item()
		}
		grid = generate_grid_image(imgs_source_for_log, imgs_target_for_log, imgs_shifted_for_log, imgs_stylemask_for_log)
		save_image(grid, os.path.join(self.log_dir,'eval_images','eval_{:06d}.png'.format(step)))
		if self.use_wandb:
			wandb.log(metrics)
			image_array = grid.detach().cpu().numpy()
			image_array = np.transpose(image_array, (1, 2, 0))
			images = wandb.Image(image_array)
			wandb.log({"eval images": images})

		self.fuser_diff.train()
		self.fuser_inner.train()
		return metrics
