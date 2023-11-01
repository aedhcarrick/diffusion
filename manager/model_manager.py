#  managers/model_manager.py


import utils.devices as _dev
import manager.utils as _utils
import logging

from utils.logger import ThreadContextFilter
from utils.path import get_path
from typing import List, Literal, Union, Tuple
from manager.models import Unet, Clip, Vae
from manager.components import Conditioning, Latent, Sampler


class ModelManager():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(ThreadContextFilter())
		self.unet = None
		self.clip = None
		self.vae = None
		self.conditioning: List[Conditioning] = []
		self.latents = None
		self.sampler = Sampler()
		self.samples = {}

	def load_model(
			self,
			model_name: str
			) -> Tuple[Union[None, Unet], Union[None, Clip], Union[None, Vae]]:
		unet, clip, vae = _utils.get_model(model_name)
		self.unet = unet
		self.clip = clip
		self.vae = vae
		return unet, clip, vae

	def get_cond(self, clip: Union[None, Clip] = None, text: str) -> Conditioning:
		if clip is None:
			clip = self.clip
		return clip.encode(text)

	def generate_empty_latents(self, size: Tuple[int,int,int]) -> Latent:
		if self.unet is None:
			log.warning('Please load a model first!')
			return
		if self.latents is None:
			self.latents = Latent(self.unet.base_type)
		self.latents.generate(size[0], size[1], size[2])

	def sample(
			self,
			model=None,
			clip=None,
			vae=None,
			latents=None,
			sampler=None,
			scheduler=None,
			cond=None,
			clip_skip=None,
			seed=42,
			cfg=7.5,
			steps=50,
			width=512,
			height=512,
			batch_size=1,
			):
		if model is None:
			if self.unet is None:
				log.warning('No model found! Please load a model first!')
				return
			else:
				model = self.unet
		if clip is None:
			clip = self.clip
		if vae is None:
			vae = self.vae
		if latents is None:
			latents = Latent(model.base_type)
			latents.generate(batch_size,height,width)
		if sampler is None:
			sampler = self.sampler
		if clip_skip is not None:
			self.clip_skip(clip, clip_skip)
		self.samples.update(
				_utils.sample(
						model,
						clip,
						vae,
						latents,
						sampler,
						scheduler,
						cond,
						seed,
						cfg,
						steps,
						width,
						height,
						batch_size
				)
		)
		images = self.vae_decode(vae, self.samples)
		for image in images:
			self.save_image(image)

	def vae_decode(self, vae, samples):
		if vae is None:
			vae = self.vae
		return vae.decode(samples)

	def vae_encode(self, vae, image):
		pass

	def clip_skip(self, clip, stop_at_clip_layer):
		pass

	def load_lora(self, model, clip, lora_name, str_model, str_clip):
		pass

	def save_images(self, images):
		output_dir = get_path('output_dir')
		base_count = len(os.listdir(output_dir))
		for image in images:
			image.save(os.path.join(output_dir, f"{base_count:05}.png")
			base_count += 1



