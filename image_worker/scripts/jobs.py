# workflow.py


import cv2
import numpy as np
import os
import sys
import time
import torch
import uuid

from contextlib import contextmanager, nullcontext
from einops import rearrange
from image_worker import scripts
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm, trange
from typing import Literal, Union


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
			setattr(self, key, value)
		os.makedirs(self.output_dir, exist_ok=True)

	def get_sampler(self, model):
		if self.sampler == "DPM":
			return DPMSolverSampler(model)
		elif self.sampler == "PLMS":
			return PLMSSampler(model)
		else:
			return DDIMSampler(model)

	def execute(self, model, device, outpath):
		sampler = self.get_sampler(model)
		seed_everything(self.seed)
		assert self.prompt is not None
		data = [self.batch_size * [self.prompt]]

		base_count = len(os.listdir(outpath))

		start_code = None
		if self.fixed_code:
			start_code = torch.randn([self.batch_size, self.channels, self.height // self.down_sampling, self.width // self.down_sampling], device=device)
		precision_scope = autocast if self.precision=="autocast" else nullcontext
		with torch.no_grad():
			with precision_scope("cuda"):
				with model.ema_scope():
					tic = time.time()
					all_samples = list()
					for n in trange(self.batch_size, desc="Sampling"):
						for prompts in tqdm(data, desc="data"):
							uc = None
							if self.guidance != 1.0:
								uc = model.get_learned_conditioning(self.batch_size * [""])
							if isinstance(prompts, tuple):
								prompts = list(prompts)
							c = model.get_learned_conditioning(prompts)
							shape = [self.channels, self.height // self.down_sampling, self.width // self.down_sampling]
							samples, _ = sampler.sample(
									S = self.steps,
									conditioning = c,
									batch_size = self.batch_size,
									shape = shape,
									verbose = False,
									unconditional_guidance_scale = self.guidance,
									unconditional_conditioning = uc,
									eta = self.eta,
									x_T = start_code
							)

							x_samples = model.decode_first_stage(samples)
							x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
							x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
							x_image = x_samples
							x_image_torch = torch.from_numpy(x_image).permute(0, 3, 1, 2)

							for x_sample in x_image_torch:
								x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
								img = Image.fromarray(x_sample.astype(np.uint8))
								img.save(os.path.join(outpath, f"{base_count:03}.png"))
								base_count += 1
					toc = time.time()
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

	def run(self, manager):
		for oper in self.operations:
			try:
				model = manager.get_loaded_model(oper.model)
				device = manager.device
				outpath = manager.output_dir
				assert oper.execute(model, device, outpath)
			except Exception as e:
				print(f'job: {self.job_ID} halted.')
				print(f'operation failed:  {e}')
				return False
		return True


