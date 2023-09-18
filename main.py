#   models.py

import os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

class Sampler():
	def __init__(self, name: Literal["DPM", "PLMS", "DDIM"]):
		self.name = name

	def get_sampler(self, model):
		if self.name == "DPM":
			return DPMSolverSampler(model)
		elif self.name == "PLMS":
			return PLMSSampler(model)
		else:
			return DDIMSampler(model)


class Options():
	def __init__(self):
		self.prompt: str = "a painting of a dog eating nachos"
		self.output_dir: str = "outputs"
		self.config: str = "configs/stable-diffusion/v1-inference.yaml"
		self.model: str = "models/checkpoints/model.ckpt"
		self.sampler: Union[None, Sampler] = None
		self.iter: int = 2
		self.steps: int = 50
		self.eta: float = 0.0
		self.height: int = 512
		self.width: int = 512
		self.channels: int = 4
		self.down_sampling: int = 8
		self.guidance: float = 7.5
		self.batch_size: int = 4
		self.seed: int = 42
		self.precision: Literal["full", "autocast"] = "autocast"

	def set_diffuser(self, diff_name: str):
		self.diffuser = Diffuser(diff_name)

	def set_sampler(self, samp_name: str):
		self.sampler = Sampler(samp_name)

def main():
	opt = Options()
	seed_everything(opt.seed)

	# get model
	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.model}")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	# get sampler
	if not opt.sampler:
		opt.sampler = Sampler('DDIM')
	sampler = opt.sampler.getSampler(model)

	# create output
	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	# get prompt
	assert opt.prompt is not None
	data = [opt.batch_size * [opt.prompt]]


	start_code = None
#	if opt.fixed_code:
#		start_code = torch.randn([opt.batch_size, opt.channels, opt.height // opt.guidance, opt.width // opt.guidance], device=device)

	precision_scope = autocast if opt.precision=="autocast" else nullcontext
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				all_samples = list()
				for n in trange(opt.batch_size, desc="Sampling"):
					for prompts in tqdm(data, desc="data"):
						uc = None
						if opt.guidance != 1.0:
							uc = model.get_learned_conditioning(opt.batch_size * [""])
						if isinstance(prompts, tuple):
							prompts = list(prompts)
						c = model.get_learned_conditioning(prompts)
						shape = [opt.channels, opt.height // opt.guidance, opt.width // opt.guidance]
						samples, _ = sampler.sample(
								S = opt.steps,
								conditioning = c,
								batch_size = opt.batch_size,
								shape = shape,
								verbose = False,
								unconditional_guidance_scale = opt.guidance,
								unconditional_conditioning = uc,
								eta = opt.eta,
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
							img.save(os.path.join(sample_path, f"{base_count:03}.png"))
							base_count += 1
				toc = time.time()

	print(f"Output saved to: \n{outpath} \n")


if __name__ == "__main__":
	main()
