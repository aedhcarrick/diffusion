# txt2img.py

import cv2
import glob
import numpy as np
import os
import sys
import time
import torch

from contextlib import contextmanager, nullcontext
from einops import rearrange
from itertools import islice
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from scripts.jobs import Options, Sampler
from torch import autocast
from tqdm import tqdm, trange


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


def txt2img(opt: Options):
	seed_everything(opt.seed)

	# get model
	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.model}")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	# get sampler
	if not opt.sampler:
		opt.sampler = Sampler('DDIM')
	sampler = opt.sampler.get_sampler(model)

	# create output
	os.makedirs(opt.output_dir, exist_ok=True)
	outpath = opt.output_dir
	base_count = len(os.listdir(outpath))

	# get prompt
	assert opt.prompt is not None
	data = [opt.batch_size * [opt.prompt]]


	start_code = None
#	if opt.fixed_code:
#		start_code = torch.randn([opt.batch_size, opt.channels, opt.height // opt.downsampling, opt.width // opt.down_sampling], device=device)

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
						shape = [opt.channels, opt.height // opt.down_sampling, opt.width // opt.down_sampling]
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
							img.save(os.path.join(outpath, f"{base_count:03}.png"))
							base_count += 1
				toc = time.time()

	return True

