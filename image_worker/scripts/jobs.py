# workflow.py


import diffusers
import numpy as np
import os
import sys
import torch
import torchvision
import uuid


from diffusers import (
		DDPMScheduler,
		EulerAncestralDiscreteScheduler,
		DPMSolverMultistepScheduler,
		DPMSolverSinglestepScheduler,
		LMSDiscreteScheduler,
		PNDMScheduler,
		DDIMScheduler,
		EulerDiscreteScheduler,
		HeunDiscreteScheduler,
		KDPM2DiscreteScheduler,
		KDPM2AncestralDiscreteScheduler,
		AutoencoderKL,
		UNet2DConditionModel
		)
from einops import rearrange
from enum import Enum
from torch import einsum
from torchvision import transforms
from tqdm.autonotebook import tqdm
from transformers import (
		CLIPTextModel,
		CLIPTokenizer
		)
from typing import Literal


class SchedulerType(Enum):
	NONE	= 'none'
	DDIM	= 'ddim'
	DDPM	= 'ddpm'
	DPM		= 'dpm'
	DPMM	= 'dpmm'
	EULER	= 'euler'
	EULERA	= 'euler_a'
	HEUN	= 'heun'
	KDPM2A	= 'kdpm2_a'
	KDPM2	= 'kdpm2'
	LMS		= 'lms'
	PNDM	= 'pndm'



class TextToImageOperation():
	def __init__(self, settings: dict):
		self.prompt: str = "a painting of a dog eating nachos"
		self.config: str = "configs/stable-diffusion/v1-inference.yaml"
		self.model: str = "model.ckpt"
		self.sampler: str = None
		self.iter: int = 2
		self.steps: int = 50
		self.eta: float = 0.0
		self.height: int = 512
		self.width: int = 512
		self.channels: int = 4
		self.down_sampling: int = 8
		self.guidance: float = 7.5
		self.batch_size: int = 1
		self.seed: int = 42
		self.fixed_code: bool = False
		self.precision: Literal["full", "autocast"] = "autocast"
		for key, value in settings.items():
			if value is not None:
				setattr(self, key, value)

	def get_scheduler(self):
		if self.sampler == SchedulerType.DDIM:
			return DDIMScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.DDPM:
			return DDPMScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.DPM:
			return DPMSolverSinglestepScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.DPMM:
			return DPMSolverMultistepScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.EULER:
			return EulerDiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.EULERA:
			return EulerAncestralDiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.HEUN:
			return HeunDiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.KDPM2:
			return KDPM2DiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.KDPM2A:
			return KDPM2AncestralDiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.LMS:
			return LMSDiscreteScheduler.from_pretrained(self.model, subfolder="scheduler")
		if self.sampler == SchedulerType.PNDM:
			return PNDMScheduler.from_pretrained(self.model, subfolder="scheduler")

	def get_vae(self):
		return AutoEncoderKL.from_pretrained(self.model, subfolder="vae", use_safetensors=True)

	def get_tokenizer(self):
		return CLIPTokenizer.from_pretrained(self.model, subfolder="tokenizer")

	def get_text_encoder(self):
		return CLIPTextModel.from_pretrained(self.model, subfolder="text_encoder", use_safetensors=True)

	def get_unet(self):
		return UNet2DConditionModel.from_pretrained(self.model, subfolder="unet", use_safetensors=True)

	def execute(self, device, input_dir, output_dir):
		# load model
		scheduler = self.get_scheduler()
		vae = self.get_vae()
		tokenizer = self.get_tokenizer()
		text_encoder = self.get_text_encoder()
		unet = self.get_unet()
		vae.to(device)
		text_encoder.to(device)
		unet.to(device)
		# create embeddings
		text_input = tokenizer(
				self.prompt,
				padding='max_length',
				max_length=tokenizer.model_max_length,
				truncation=True,
				return_tensors="pt"
				)
		with torch.no_grad():
			text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
		max_length = text_input.input_ids.shape[-1]
		uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
		uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
		text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
		# create latent image
		latent_images = torch.randn(
				(self.batch_size, unet.in_channels, self.height // self.down_sampling, self.width // self.down_sampling),
				generator=torch.manual_seed(0),
				)
		latent_images = latent_images.to(device)
		# denoise image
		latent_images = latent_images * scheduler.init_noise_sigma
		scheduler.set_timesteps(self.steps)
		for t in tqdm(scheduler.timesteps):
			# expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
			latent_model_input = torch.cat([latent_images] * 2)
			latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
			# predict the noise residual
			with torch.no_grad():
				noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
			# perform guidance
			noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
			noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
			# compute the previous noisy sample x_t -> x_t-1
			latent_images = scheduler.step(noise_pred, t, latent_images).prev_sample
		# decode image
		latent_images = 1 / 0.18215 * latent_images
		with torch.no_grad():
			image = vae.decode(latent_images).sample
		# convert and save image
		image = (image / 2 + 0.5).clamp(0, 1).squeeze()
		image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
		images = (image * 255).round().astype("uint8")
		image = Image.fromarray(image)
		image.save(os.path.join(output_dir, f"{base_count:03}.png"))
		return True


class ImageJob():
	def __init__(self, settings: dict):
		self.job_ID: str = str(uuid.uuid4())
		self.job_type = settings['job_type']
		self.operations: list = []
		for op in settings['operations']:
			if op['oper_type'] == 'txt2img':
				oper = TextToImageOperation(op)
			self.operations.append(oper)

	def run(self, manager, input_dir, output_dir):
		success = True
		for oper in self.operations:
#			model = manager.get_loaded_model(oper.model)
			device = manager.device
			if not oper.execute(device, input_dir, output_dir):
				success = False
		return success


