#  managers/model_manager.py


import device_manager as dm
from utils.logger import Logger, ThreadContextFilter


class ModelManager():
	log = Logger(".".join([__name__, self.__class__.__name__]))
	log.addFilter(ThreadContextFilter())
	vram_state = dm.vram_state
	directml_enabled = dm.directml_enabled
	torch_device = dm.torch_device
	unet_offload_device = dm.unet_offload_device
	unet_load_device = dm.unet_load_device
	text_encoder_offload_device = dm.text_encoder_offload_device
	text_encoder_device = dm.text_encoder_device
	vae_offload_device = dm.vae_offload_device
	vae_load_device = dm.vae_load_device

	def __init__(self, input_dir='input', output_dir='output'):
		self.input = input_dir
		self.output = output_dir
		self.paths = {}
		self.loaded_models = []

	def load_model(self, model_name):
		pass

	def text_encode(self, clip, text):
		pass

	def generate_empty_latent(self, width, height, batch_size=1):
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






