import os
import numpy as np

hyper_model_arguments = {
	'image_resolution':				256,
	'channel_multiplier':			1,
	'generator_weights':		    './pretrained_models/stylegan-voxceleb.pt',

	'deca_layer':					'layer4',
	'arcface_layer':				23,
	'hypernet_type':				'SharedWeightsHyperNetResNet',
	'use_truncation':				True,
	'layers_to_tune':				'0,2,3,5,6,8,9,11,12,14,15,17,18',
	'kernel_size':					1,
	'pretrained_pose_encoder':		'Deca',
	'pretrained_app_encoder':		'ArcFace',
	'mode':							'delta_per_channel',

	'pose_encoder_path':			'./libs/DECA/data/deca_model.tar', 
	'app_encoder_path':				'./pretrained_models/insight_face.pth', 
	'source_e4e_path':						'./pretrained_models/encoder_mean_fine.pt',
	'target_e4e_path':						'./pretrained_models/encoder_mean_fine.pt',
	'fuser_path':						None,
	'mask_net_path':						'./pretrained_models/mask_net_pretrained.pt',
	'sfd_detector_path':			'./pretrained_models/s3fd-619a316812.pth',
    'split_sections':			[512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64],

}

stylegan2_ffhq_1024 = {
	'image_resolution':			1024,
	'channel_multiplier':		2,
	'gan_weights':				'./pretrained_models/stylegan-voxceleb.pt',

	'stylespace_dim':			6048,
	'split_sections':			[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32],

	'e4e_inversion_model':		'./pretrained_models/e4e_ffhq_encode_1024.pt',
	'expression_ranges':		'./libs/configs/ranges_FFHQ.npy' # Used for evaluation
}



