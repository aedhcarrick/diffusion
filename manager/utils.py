#  manager/utils.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


from ldm.util import instantiate_from_config
from manager.models import Unet, Clip, Vae
from omegaconf import OmegaConf
from typing import List, Literal, Optional, Tuple, Union
from utils.devices import unet_dtype
from utils.paths import get_full_path


import torch, safetensors


def get_state_dict(ckpt, device=None):
	if device is None:
		device = torch.device("cpu")
	if ckpt.lower().endswith(".safetensors"):
		sd = safetensors.torch.load_file(ckpt, device=device.type)
	else:
		pl_sd = torch.load(ckpt, map_location=device)
		if "state_dict" in pl_sd:
			sd = pl_sd["state_dict"]
		else:
			sd = pl_sd
	return sd

def get_model_base_type(state_dict: dict) -> Literal["sd1", "sd2", "sdxl"]:
	model_type = None
	## this function needs work
	for key in state_dict.keys():
		if key.startswith("cond_stage_model.transformer.text_model"):
			model_type = "sd1"
			break
		elif key.startswith("cond_stage_model.model.text_model"):
			model_type = "sd2"
			break
		elif key.startswith("conditioner"):  # <--this should work?
			model_type = "sdxl"
			break
	if model_type is None:
		log.error("Could not determine the base type of the diffusion model.")
	return model_type

def get_model_config(state_dict: dict, base_type: Literal["sd1", "sd2", "sdxl"]) -> OmegaConf:
	if base_type == 'sd1':
		return OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
	if base_type == 'sd2':
		return OmegaConf.load('configs/stable-diffusion/v2-inference.yaml')
	log.error(f'{base_type}-based models are not yet supported.')
	return None

def load_model_from_config(ckpt, config, state_dict):
		model = instantiate_from_config(config.model)
		# for debug
		print('model keys:')
		print(model.__dict__.keys())
		print('state_dict keys')
		print(state_dict.keys())
		m, u = model.load_state_dict(state_dict, strict=False)
		if len(m) > 0:
			log.error(f"missing keys:  {m}")
		if len(u) > 0:
			log.error(f"unexpected keys:  {u}")
		if len(m) + len(u) == 0:
			log.info(f'All keys matched.')
		return model, m, u

def get_model(
			model_name: str,
			config: Union[None, str] = None,
			get_clip: Optional[bool] = True,
			get_vae: Optional[bool] = True,
			) -> Tuple[Union[None, Unet], Union[None, Clip], Union[None, Vae]]:
	full_path = get_full_path('checkpoints', model_name)
	if not full_path:
		log.info(f'Failed to find model: {model_name}')
		return (None, None, None)
	state_dict = get_state_dict(full_path)
	base_type = get_model_base_type(state_dict)
	if config is None:
		config = get_model_config(state_dict, base_type)
	model, m, u = load_model_from_config(full_path, config, state_dict)
	if base_type == 'sd1':
		model_config = config['model']['params']
		unet = Unet(
				name=model_name,
				model=model.get_submodule('model.diffusion_model'),
				params=model_config['unet_config'],
				base_type=base_type
		)
		get_clip = get_clip
		get_vae = get_vae
		for key in m:
			if key.startswith('cond_stage_model'):
				log.warning(f'Clip not found in checkpoint: {model_name}')
				get_clip = False
			if key.startswith('first_stage_model'):
				log.warning(f'Vae not found in checkpoint: {model_name}')
				get_vae = False
				break
		if get_clip:
			clip = Clip(
					name=model_name,
					model=model.cond_stage_model,
					params=model_config['cond_stage_config'],
					base_type=base_type
			)
		else:
			clip = None
		if get_vae:
			vae = Vae(
					name=model_name,
					model=model.first_stage_model,
					params=model_config['first_stage_config'],
					base_type=base_type
			)
		else:
			vae = None
	else:
		raise ValueError("Unsupported model base type.")
		log.error(f'Failed to get model: {model_name}')
	return unet, clip, vae

def sample(
		model,
		clip,
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
		):
	if not model.is_loaded():
		model.load()
	if not clip.is_loaded():
		clip.load()
	if latents is None:
		latents = Latent(model.base_type)
		latents.generate(batch_size,height,width)

    data = [batch_size * [cond.prompts]]
	output = []
	with torch.no_grad():
		with model.ema_scope():
			all_samples = list()
			for n in trange(batch_size, desc="Sampling"):
				for prompts in tqdm(data, desc="data"):
					uc = None
					if cond is not None and cfg != 1.0:
						uc = clip.get_learned_conditioning(batch_size * [""])
					if isinstance(prompts, tuple):
						prompts = list(prompts)
					c = clip.get_learned_conditioning(prompts)
					shape = latent.samples[n]

					samples, _ = sampler.sample(
							S=steps,
							conditioning=c,
							batch_size=batch_size,
							shape=shape,
							verbose=False,
							unconditional_guidance_scale=cfg,
							unconditional_conditioning=uc,
							eta=0.0,
							x_T=None
					)
					output.append(samples)
	return output




