#  managers/model_manager.py


import utils.devices as _dev
import manager.utils as _utils
import logging

from utils.logger import ThreadContextFilter
from typing import List, Literal, Union, Tuple
from manager.models import Unet, Clip, Vae
from manager.components import Conditioning, Latent


class ModelManager():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(ThreadContextFilter())
		self.model = None
		self.clip = None
		self.vae = None
		self.conditioning = []
		self.latents = []

	def load_model(self, model_name: str) -> Tuple[Unet, Clip, Vae]:
		unet, clip, vae = _utils.get_model(model_name)
		self.model = unet
		self.clip = clip
		self.vae = vae
		return unet, clip, vae

	def text_encode(self, clip: Clip, text: str) -> Conditioning:
		pass

	def generate_empty_latents(self, size: Tuple[int,int,int]) -> List[Latent]:
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




