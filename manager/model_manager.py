#  managers/model_manager.py


import utils.devices as _dev
import manager.utils as _utils
import logging

from utils.logger import ThreadContextFilter
from utils.path import get_path
from typing import List, Literal, Union, Tuple
from manager.models import Unet, Clip, Vae, LoadedModel
from manager.components import Conditioning, Latent, Sampler, Scheduler


class ModelManager():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(ThreadContextFilter())
		self.loaded_models = []

	def get_loaded_model(self, model_name: str) -> LoadedModel:
		for loaded_model in self.loaded_models:
			if model_name == loaded_model.name:
				index = self.loaded_models.index(loaded_model)
				return self.loaded_models[index]
		loaded_model = _utils.get_model(model_name)
		self.loaded_models.append(loaded_model)
		return loaded_model

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		pass

	def load_clip(self, model_name):
		pass

	def load_tokenizer(self):
		pass

	def load_vae(self):
		pass

	def txt2img(
			self,
			model: LoadedModel,
			prompts: Union[str, List[str]],
			clip_skip=None,
			sampler=None,
			scheduler=None,
			seed=42,
			cfg=7.5,
			steps=50,
			width=512,
			height=512,
			batch_size=1,
			):
		return _utils.txt2img(
				model,
				prompts,
				clip_skip,
				sampler,
				scheduler,
				seed,
				cfg,
				steps,
				width,
				height,
				batch_size)

	def save_images(self, images):
		output_dir = get_path('output_dir')
		base_count = len(os.listdir(output_dir))
		for image in images:
			image.save(os.path.join(output_dir, f"{base_count:05}.png")
			base_count += 1




