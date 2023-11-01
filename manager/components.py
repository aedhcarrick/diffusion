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
		base_type: Literal['sd1', 'sd2', 'sdxl'] = base_type
		scale_factor = 1.0
		samples: Union[None, torch.Tensor] = None

	def generate(self, batch_size: int, width: int, height: int):
		if batch_size <= 0:
			log.warning('Batch size must be 1 or greater!')
		if (width / 8
		self.samples = torch.zeros([batch_size, 4, height // 8, width // 8])


class Sampler():
	def __init__(self):
		self.model = None
		self.conditioning = None
		self.latents = None

	def sample(self, final_model, conditioning, latents, seed, start, end):
		pass






