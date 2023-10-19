#  model_manager.py

#
#	Adapted from 'https://github.com/comfyanonymous/ComfyUI'.
#	--> GNU GPLv3
#

import io
import logging
import os
import safetensors
import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from models import ModelConfig, ModelType
from models import BaseModel, LoadedModel, SD2_1_UNCLIP
from omegaconf import OmegaConf
from utils.cli_args import args
from utils.log_config import ThreadContextFilter
from utils import device_management as _utils


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


def get_model_config(model_name, model_path, state_dict=None):
	log.info(f'Loading configs..')
	config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
	params = config['model']['params']
	unet_config = {}
	unet_extra_config = {
		"num_heads": -1,
		"num_head_channels": 64,
	}
	dtype = torch.float32
	if "unet_config" in params:
		if "params" in params['unet_config']:
			unet_config = config_params["unet_config"]["params"]
			if "use_fp16" in unet_config:
				if unet_config["use_fp16"]:
					unet_config["dtype"] = torch.float16
					dtype = torch.float16
			for x in unet_extra_config:
				unet_config[x] = unet_extra_config[x]

	noise_aug_config = None
	if "noise_aug_config" in params:
		noise_aug_config = params["noise_aug_config"]

	model_type = ModelType.EPS
	if "parameterization" in params:
		if params["parameterization"] == "v":
			model_type = ModelType.V_PREDICTION

	scale_factor = params['scale_factor']

	model_config = ModelConfig(model_name, model_path)
	model_config.unet_config = unet_config
	model_config.vae_config = params['first_stage_config']
	model_config.clip_config = params['cond_stage_config']
	model_config.noise_aug_config = noise_aug_config
	model_config.model_type = model_type
	model_config.dtype = dtype
	return model_config

def get_model_from_config(model_config):
	if model_config.noise_aug_config is not None:
		model = SD2_1_UNCLIP(model_config)
	else:
		model = BaseModel(model_config)

	if model_config.is_inpaint():
		model.set_inpaint()

	if model_config.dtype == torch.float16:  # might want to double check this
		model = model.half()

	offload_device = _utils.get_unet_offload_device()
	return LoadedModel(model, load_device=_utils.get_torch_device(), offload_device=offload_device)


class ModelManager():
	def __init__(self, model_dir):
		log.info('Initializing model manager..')
		self.available_models = []
		self.loaded_models = []
		self.offload_device = torch.device('cpu')
		self.load_device = _utils.get_torch_device()
		self.model_dir = model_dir
		self.ckpt_dir = os.path.join(model_dir, 'checkpoints')
		self.get_available_models()
		self.dtype = torch.float16

	def get_available_models(self):
		log.info('Getting available models..')
		self.available_models.clear()
		for f in os.listdir(self.ckpt_dir):
			if f.endswith('.ckpt') or f.endswith('.safetensors'):
				self.available_models.append(f)
				log.info(f'    {f}')

	def load_model(self, model_name):
		model_path = os.path.join(self.ckpt_dir, model_name)
		model_config = get_model_config(model_name, model_path)
		loaded_model = get_model_from_config(model_config)
		self.loaded_models.append(loaded_model)
		return loaded_model

	def get_loaded_model(self, model_name):
		log.info('Getting model..')
		for loaded_model in self.loaded_models:
			if loaded_model.name == model_name:
				log.info('Model already loaded.')
				return loaded_model
		for model in self.available_models:
			if model == model_name:
				log.info('Model not yet loaded.')
				return self.load_model(model)
		log.error(f'Model not found:  {model_name}')

	def get_sampler(self, sampler_name, model):
		log.info(f'Loading sampler "{sampler_name}"..')
		if sampler_name == "DPM":
			return DPMSolverSampler(model)
		elif sampler_name == "PLMS":
			return PLMSSampler(model)
		else:
			return DDIMSampler(model)

	def




