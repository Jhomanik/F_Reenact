import os
import datetime
import random
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
import torch
torch.cuda.set_device(5)
from libs.trainer import Trainer
from libs.old_trainer import OldTrainer

		
def main():
	"""
	Training script.
	Options:
		######### General ###########
		--experiment_path		   			: path to save experiment
		--use_wandb		   		   			: use wandb to log losses and evaluation metrics
		--log_images_wandb	       			: if True log images on wandb	
		--project_wandb            			: Project name for wandb
		--resume_training_model				: Path to model to continue training or None

		######### Generator #########
		--dataset_type						: voxceleb or ffhq
		--image_resolution         			: image resolution of pre-trained GAN model. image resolution for voxceleb dataset is 256

		######### Dataset #########
		--synthetic_dataset_path        	: set synthetic dataset path for evaluation. npy file with random synthetic latent codes.
		
		######### Training #########
		--lr								: set the learning rate of direction matrix model
		--num_layers_control				: number of layers to apply the mask
		--max_iter	                		: set maximum number of training iterations
		--batch_size		   	   			: set training batch size

		--lambda_identity					: identity loss weight
		--lambda_perceptual				    : perceptual loss weight
		--lambda_shape						: shape loss weight
		--use_recurrent_cycle_loss		    : use recurrent cycle loss
		
		--steps_per_log	                	: set number iterations per log
		--steps_per_save_models		   	   	: set number iterations per saving model
		--steps_per_evaluation				: set number iterations per model evaluation during training
		--validation_pairs					: number of validation pairs for evaluation
		--num_pairs_log						: number of pairs to visualize during evaluation

		######################

		python run_trainer.py --experiment_path ./training_attempts/exp_v00 
	"""
	parser = argparse.ArgumentParser(description="training script")

	######### General ###########
	parser.add_argument('--experiment_path', type=str, required = True, help="path to save the experiment")
	parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', help="use wandb to log losses and evaluation metrics")
	parser.set_defaults(use_wandb=False)
	parser.add_argument('--log_images_wandb', dest='log_images_wandb', action='store_true', help="if True log images on wandb")
	parser.set_defaults(log_images_wandb=False)
	parser.add_argument('--project_wandb', type=str, default = 'stylespace', help="Project name for wandb")


	######### Generator #########
	parser.add_argument('--image_resolution', type=int, default=256, choices=(256, 1024), help="image resolution of pre-trained GAN modeln")
	parser.add_argument('--type', type=str, default='rec', help="set dataset name")

	######### Dataset #########
	parser.add_argument('--train_dataset_path', type=str, default=None, help="set dataset path for train")
	parser.add_argument('--test_dataset_path', type=str, default=None, help="set dataset path for evaluation")
	parser.add_argument('--cherry_dataset_path', type=str, default=None, help="set  path for cheery")
	######### Training #########
	parser.add_argument('--lr', type=float, default=0.0001, help=" set the learning rate of direction matrix model")
	parser.add_argument('--epochs', type=int, default=3, help=" set number of epochs")
	parser.add_argument('--num_layers_control', type=int, default=10, help="setnumber of layers to apply the mask")
	parser.add_argument('--max_iter', type=int, default=1000000, help="set maximum number of training iterations")
	parser.add_argument('--batch_size', type=int, default=3, help="set training batch size")
	parser.add_argument('--test_batch_size', type=int, default=4, help="set test batch size")
	#Hyper coefs, pixel-wise seperately, iters




	parser.add_argument('--lambda_identity', type=float, default=0.1, help="")
	parser.add_argument('--lambda_perceptual', type=float, default=0.8, help="")
	parser.add_argument('--lambda_shape', type=float, default=0, help="")
	parser.add_argument('--lambda_pixel', type=float, default=1.0, help="")




	parser.add_argument('--validation_pairs', type =int, default=1500, help="valid pairs")
	parser.add_argument('--steps_per_log', type=int, default=5, help="print log")
	parser.add_argument('--steps_per_image_log', type=int, default=600, help="print image log")
	parser.add_argument('--steps_per_save_models', type=int, default=600, help="steps per save model")
	parser.add_argument('--steps_per_evaluation', type=int, default=600, help="steps per evaluation during training")

	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	# Create output dir and save current arguments
	experiment_path = args['experiment_path']
	experiment_path = experiment_path + '_{}_{}'.format(args['type'], args['image_resolution'])
	args['experiment_path'] = experiment_path
	# Set up trainer
	print("#. Experiment: {}".format(experiment_path))


	trainer = Trainer(args)
	trainer.train()
	

if __name__ == '__main__':
	main()


	
