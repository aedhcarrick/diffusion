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


from transformers import CLIPConfig, CLIPModel


class TextEncoder(torch.nn.module):
	pass


class SD1TextEncoder(TextEncoder):
	pass


class SD2TextEncoder(TextEncoder):
	pass


class SDXLTextEncoder(TextEncoder):
	pass


def getTextEncoder(base_type: Literal['sd1', 'sd2', 'sdxl']) -> TextEncoder:
	if base_type == 'sd1':
		return SD1TextEncoder()
	elif base_type == 'sd2':
		return SD2TextEncoder()
	elif base_type == 'sdxl':
		return SDXLTextEncoder()


class FirstStageModel(torch.nn.module):
	def __init__(self, state_dict, config):
		if config is None:
			ddconfig = {
				'double_z': True,
				'z_channels': 4,
				'resolution': 256,
				'in_channels': 3,
				'out_ch': 3,
				'ch': 128,
				'ch_mult': [1, 2, 4, 4],
				'num_res_blocks': 2,
				'attn_resolutions': [],
				'dropout': 0.0
			}
			self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
		else:
			self.first_stage_model = AutoencoderKL(**(config))
		self.first_stage_model.eval()
		m, u = self.first_stage_model.load_state_dict(state_dict, strict=False)
		if len(m) > 0:
			print("Missing VAE keys", m)
		if len(u) > 0:
			print("Leftover VAE keys", u)


class SD1FirstStageModel(FirstStageModel):
	def __init__(self, config):
		super().__init__(config)


class SD2FirstStageModel(FirstStageModel):
	def __init__(self, config):
		super().__init__(config)


class SDXLFirstStageModel(FirstStageModel):
	def __init__(self, config):
		super().__init__(config)


def getFirstStageModel(base_type: Literal['sd1', 'sd2', 'sdxl']) -> FirstStageModel:
	if base_type == 'sd1':
		return SD1FirstStageModel()
	elif base_type == 'sd2':
		return SD2FirstStageModel()
	elif base_type == 'sdxl':
		return SDXLFirstStageModel()


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
	def __init__(self, base_type, scale_factor):
		self.base_type: Literal['sd1', 'sd2', 'sdxl'] = base_type
		self.down_sampling = scale_factor
		self.latent_channels = 4

	def generate(self, batch_size: int, width: int, height: int):
		if batch_size <= 0:
			raise ValueError('Batch size must be 1 or greater!')
		if (width % 8 != 0) or (height % 8 != 0):
			raise ValueError(f"height and width must be a multiple of 8!")
		return torch.zeros(
				[
					batch_size,
					self.latent_channels,
					height // self.down_sampling,
					width // self.down_sampling
				]
				)


import k_diffusion


class Sampler():
	def __init__(self, model, sampler):
		self.name = sampler
		self.model = model
		self.wrapper = k_diffusion.external.ComVizDenoiser(model)
		self.denoiser = CFGDenoiser(self.wrapper)
		self.sampling = k_diffusion.sampling.__dict__[f'sample_{self.name}']

	def sample(
			self,
			steps,
			cond,
			latents,
			cfg,
			batch_size
			eta,
			x_T
			)
		sigmas = self.wrapper.get_sigmas(steps)
		x = x_T * sigmas[0]
		data = [batch_size * [prompts]]
		samples = []
		for prompt in tqdm(data, desc="data"):
			uc = None
			if prompts is not None and cfg != 1.0:
				uc = clip.get_learned_conditioning(batch_size * [""])
			if isinstance(prompts, tuple):
				prompts = list(prompts)
			c = clip.get_learned_conditioning(prompts)
			sample, _ = self.sampling(
					S=steps,
					conditioning=c,
					batch_size=batch_size,
					shape=latent,
					verbose=False,
					unconditional_guidance_scale=cfg,
					unconditional_conditioning=uc,
					eta=0.0,
					x_T=None
			)
			samples.append(sample)
		return samples


from diffusers import (
	AutoencoderKL,
	DDIMScheduler,
	DPMSolverMultistepScheduler,
	EulerAncestralDiscreteScheduler,
	EulerDiscreteSchedular,
	LMSDiscreteScheduler,
	PNDMScheduler
)


class Scheduler():
	def __init__(self):




