#  model_manager.py

import torch

from ldm.util import instantiate_from_config

class LoadedModel():
	def __init__(self, name, config, device, model):
		self.name = name
		self.config = config
		self.device = device
		self.model = model


class ModelManager():
	def __init__(self, input_dir, output_dir, model_dir):
		self.log = logger.getlogger("Model Manager")
		self.available_models = []
		self.loaded_models = []
		self.device = torch.device(torch.cuda.current_device())
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.model_dir = model_dir
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
		if len(m) > 0 and verbose:
			print(f"missing keys:  {m}")
		if len(u) > 0 and verbose:
			print(f"unexpected keys:  {u}")

		model.cuda()
		model.eval()
		return model

	def get_model(self, model_name):
		for loaded_model in loaded_models:
			if loaded_model.name == model_name:
				return loaded_model.model
		for model in available_models:
			if model == model_name:
				return self.load_model(model)

	def load_model(self, model_name):
		config = 'configs/stable-diffusion/v1-inference.yaml'
		path_to_model = os.path.join(self.model_dir, model_name)
		model = self.load_model_from_config(config, path_to_model)
		loaded_model = LoadedModel(model_name, config, self.device, model)
		self.loaded_models.append(loaded_model)
		return model



