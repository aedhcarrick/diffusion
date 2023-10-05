#  model_manager.py


import logging
import os
import torch

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class LoadedModel():
	def __init__(self, name, config, device, model):
		self.name = name
		self.config = config
		self.device = device
		self.model = model


class ModelManager():
	def __init__(self, model_dir):
		self.log = logging.getLogger("Model Manager")
		self.available_models = []
		self.loaded_models = []
		self.device = torch.device(torch.cuda.current_device())
		self.model_dir = os.path.join(model_dir, 'checkpoints')
		self.get_available_models()

	def get_available_models(self):
		self.available_models.clear()
		for f in os.listdir(self.model_dir):
			if f.endswith('.ckpt') or f.endswith('.safetensors'):
				self.available_models.append(f)

	def load_model_from_config(self, config, ckpt):
		pl_sd = torch.load(ckpt, map_location="cpu")
		sd = pl_sd["state_dict"]
		model = instantiate_from_config(config.model)
		m, u = model.load_state_dict(sd, strict=False)
		if len(m) > 0:
			self.log.error(f"missing keys:  {m}")
		if len(u) > 0:
			self.log.error(f"unexpected keys:  {u}")

		model.cuda()
		model.eval()
		model = model.to(self.device)
		return model

	def get_loaded_model(self, model_name):
		for loaded_model in self.loaded_models:
			if loaded_model.name == model_name:
				return loaded_model.model
		for model in self.available_models:
			if model == model_name:
				return self.load_model(model)

	def load_model(self, model_name):
		config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
		path_to_model = os.path.join(self.model_dir, model_name)
		model = self.load_model_from_config(config, path_to_model)
		loaded_model = LoadedModel(model_name, config, self.device, model)
		self.loaded_models.append(loaded_model)
		return model



