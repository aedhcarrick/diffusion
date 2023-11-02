#  manager/components.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


from typing import List, Literal, Tuple, Union


class Tokenizer():
	def tokenize_with_weights(self, text: str):
		return text


class SD1Tokenizer(Tokenizer):
	pass


class SD2Tokenizer(Tokenizer):
	pass


class SDXLTokenizer(Tokenizer):
	pass


def getTokenizer(base_type: Literal['sd1', 'sd2', 'sdxl']) -> Tokenizer:
	if base_type == 'sd1':
		return SD1Tokenizer()
	elif base_type == 'sd1':
		return SD2Tokenizer()
	elif base_type == 'sdxl':
		return SDXLTokenizer()
	else:
		raise ValueError("Unknown diffusion model base type. Unable to supply tokenizer.")


class Conditioning():
	def __init__(
			self,
			models: List=[],
			positive: Union[None, str]=None,
			negative: Union[None, str]=None
			):
		self.models = []
		self.positive = positive
		self.negative = negative
		self.prompts = ""

	def concat(self):
		pass

	def combine(self):
		pass


import torch


class Latent():
	def __init__(self, base_type):
		self.base_type: Literal['sd1', 'sd2', 'sdxl'] = base_type
		self.down_sampling = 8
		self.latent_channels = 4
		self.samples: Union[None, torch.Tensor] = None

	def generate(self, batch_size: int, width: int, height: int):
		if batch_size <= 0:
			raise ValueError('Batch size must be 1 or greater!')
		if (width % 8 != 0) or (height % 8 != 0):
			raise ValueError(f"height and width must be a multiple of 8!")
		self.samples = torch.zeros(
				[
					batch_size,
					self.latent_channels,
					height // self.down_sampling,
					width // self.down_sampling
				]
		)


from diffusers import (
	AutoencoderKL,
	DDIMScheduler,
	DPMSolverMultistepScheduler,
	EulerAncestralDiscreteScheduler,
	EulerDiscreteSchedular,
	LMSDiscreteScheduler,
	PNDMScheduler
)


class Sampler():
	def __init__(self):
		self.model = None
		self.conditioning = None
		self.latents = None

	def sample(self, final_model, conditioning, latents, seed, start, end):
		pass


class Scheduler():
	def __init__(self):




