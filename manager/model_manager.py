#  managers/model_manager.py


import utils.devices as _dev
import utils.model_utils as _utils
import logging

from utils.logger import ThreadContextFilter
from typing import List, Tuple


class ModelManager():
	log = getLogger(".".join([__name__, self.__class__.__name__]))
	log.addFilter(ThreadContextFilter())
	vram_state = _dev.vram_state
	directml_enabled = _dev.directml_enabled
	torch_device = _dev.torch_device
	unet_offload_device = _dev.unet_offload_device
	unet_load_device = _dev.unet_load_device
	text_encoder_offload_device = _dev.text_encoder_offload_device
	text_encoder_device = _dev.text_encoder_device
	vae_offload_device = _dev.vae_offload_device
	vae_load_device = _dev.vae_load_device

	def __init__(self):
		self.paths = {}
		self.loaded_models = []

	def load_model(self, model_name: str) -> Tuple[Unet, Clip, Vae]:
		return _utils.get_model(model_name)

	def text_encode(self, clip: Clip, text: str) -> Conditioning:
		pass

	def generate_empty_latent(self, size: Tuple[int,int,int]) -> List[Latent]:
		pass

	def run_sampler(self, model, latents, sampler, scheduler, settings):
		pass

	def vae_decode(self, vae, samples):
		pass

	def vae_encode(self, vae, image):
		pass

	def clip_skip(self, clip, stop_at_clip_layer):
		pass

	def load_lora(self, model, clip, lora_name, str_model, str_clip):
		pass

	def save_image(self, image):
		pass

	#	internal
	def load_models_to_gpu(self, models, mem_required=0):
		inference_mem = pow(1024, 3)
		extra_mem = max(inference_mem, mem_required)
		to_load = []
		loaded = []
		for m in models:
			loaded_model = LoadedModel(m)
			if loaded_model in self.loaded_models:
				loaded.append(loaded_model)
			else:
				to_load.append(loaded_model)
		if len(to_load) == 0:
			devs = set(map(lambda m: m.load_device, loaded))
			for d in devs:
				if d != torch.device('cpu'):
					dm.free_memory(extra_mem, d, loaded)
			return

		total_mem_required = {}


class LoadedModel():
	def __init__(self, model):
		self.model = model
		#	state
		self.accelerated = False
		self.current_device = None
		self.load_device = None
		self.offload_device = None





