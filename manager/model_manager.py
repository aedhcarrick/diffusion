#  managers/model_manager.py


import utils.devices as _dev
import manager.utils as _utils
import logging

from utils.logger import ThreadContextFilter
from utils.path import get_path
from typing import List, Literal, Union, Tuple
from manager.models import Unet, Clip, Vae
from manager.components import Conditioning, Latent, Sampler, Scheduler


class ModelManager():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(ThreadContextFilter())
		self.unet = None
		self.clip = None
		self.vae = None
		self.prompt = ""
		self.latents = None
		self.sampler = Sampler()
		self.scheduler = Scheduler()
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

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		pass

	def load_clip(self):
		pass

	def load_tokenizer(self):
		pass

	def load_vae(self):
		pass

	def generate_empty_latents(self, size: Tuple[int,int,int]) -> Latent:
		if self.unet is None:
			raise ValueError('Please load a model first!')
			return
		if self.latents is None:
			self.latents = Latent(self.unet.base_type)
		self.latents.generate(size[0], size[1], size[2], vae.scale_factor)

	def encode_prompt(self, prompt: str) -> Conditioning:
		if clip is None:
			if self.clip is not None:
				clip = self.clip
			else:
				raise ValueError(f'Clip model and tokenizer not found! Please load first!')
		return clip.encode(text)

	def save_images(self, images):
		output_dir = get_path('output_dir')
		base_count = len(os.listdir(output_dir))
		for image in images:
			image.save(os.path.join(output_dir, f"{base_count:05}.png")
			base_count += 1

	def txt2img(
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
			self.clip.skip(clip_skip)
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
		images = self.vae.decode(vae, self.samples)
		self.save_images(image)
		log.info(f'Images finished!')



