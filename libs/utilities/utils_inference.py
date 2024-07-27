import os
import numpy as np
import torch
from torchvision import utils as torch_utils
import cv2
from skimage import io
from libs.utilities.utils import *
from libs.utilities.image_utils import read_image_opencv, torch_image_resize


def calculate_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source, imgs_target, id_loss, lpips_loss, exp_ranges):

	############ Evaluation ############
	yaw_reenacted = angles_shifted[:,0][0].detach().cpu().numpy() 
	pitch_reenacted = angles_shifted[:,1][0].detach().cpu().numpy() 
	roll_reenacted = angles_shifted[:,2][0].detach().cpu().numpy()
	exp_reenacted = params_shifted['alpha_exp'][0].detach().cpu().numpy() 
	jaw_reenacted = params_shifted['pose'][0, 3].detach().cpu().numpy() 
	
	yaw_target = angles_target[:,0][0].detach().cpu().numpy() 
	pitch_target = angles_target[:,1][0].detach().cpu().numpy() 
	roll_target = angles_target[:,2][0].detach().cpu().numpy()
	exp_target = params_target['alpha_exp'][0].detach().cpu().numpy() 
	jaw_target = params_target['pose'][0, 3].detach().cpu().numpy()

	exp_error = []	
	num_expressions = 20
	max_range = exp_ranges[3][1]
	min_range = exp_ranges[3][0]		
	jaw_target = (jaw_target - min_range)/(max_range-min_range)
	jaw_reenacted = (jaw_reenacted - min_range)/(max_range-min_range)
	exp_error.append(abs(jaw_reenacted - jaw_target))			
	for j  in range(num_expressions):
		max_range = exp_ranges[j+4][1]
		min_range = exp_ranges[j+4][0]
		target = (exp_target[j] - min_range)/(max_range-min_range)
		reenacted = (exp_reenacted[j] - min_range)/(max_range-min_range)
		exp_error.append(abs(reenacted - target) )
	exp_error = np.mean(exp_error)

	pose = (abs(yaw_reenacted-yaw_target) + abs(pitch_reenacted-pitch_target) + abs(roll_reenacted-roll_target))/3
	################################################
	###### CSIM ######
	loss_identity = id_loss(imgs_shifted, imgs_source) 
	csim = 1 - loss_identity.data.item(); imgs_shifted_255 = torch_range_1_to_255(imgs_shifted); imgs_target_255 = torch_range_1_to_255(imgs_target); lpips =  lpips_loss(imgs_shifted_255, imgs_target_255.detach()).data.item()
	return csim, pose, exp_error, lpips		




" Invert real image into the latent space of StyleGAN2 "
def invert_image(image, encoder, generator, truncation, trunc, save_path = None, save_name = None):
	with torch.no_grad():
		latent_codes = encoder(image)
		inverted_images, _ = generator([latent_codes], input_is_latent=True, return_latents = False, truncation= truncation, truncation_latent=trunc)

	if save_path is not None and save_name is not None:
		grid = torch_utils.save_image(
						inverted_images,
						os.path.join(save_path, '{}.png'.format(save_name)),
						normalize=True,
						range=(-1, 1),
					)
		# Latent code
		latent_code = latent_codes[0].detach().cpu().numpy()
		save_dir = os.path.join(save_path, '{}.npy'.format(save_name))
		np.save(save_dir, latent_code)

	return inverted_images, latent_codes