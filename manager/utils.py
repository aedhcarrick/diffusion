#  manager/utils.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


from ldm.util import instantiate_from_config
from manager.models import Unet, Clip, Vae, LoadedModel
from manager.components import Sampler
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
			) -> LoadedModel:
	full_path = get_full_path('checkpoints', model_name)
	if not full_path:
		log.info(f'Failed to find model: {model_name}')
		return (None)
	state_dict = get_state_dict(full_path)
	base_type = get_model_base_type(state_dict)
	if config is None:
	config = get_model_config(state_dict, base_type)
	model, m, u = load_model_from_config(full_path, config, state_dict)
	model_config = config['model']['params']
	clip_config = model_config.get('cond_stage_config', None)
	scale_factor = model_config.get('scale_factor', None)
	vae_config = model_config.get('first_stage_config', None)

	fp16 = False
	if 'unet_config' in model_config and 'params' in model_config['unet_config']:
		unet_config = model_config['unet_config']['params']
		if 'use_fp16' in unet_config:
			fp16 = unet_config['use_fp16']
			if fp16:
				unet_config['dtype'] = torch.float16

	noise_aug_config = None
	if 'noise_aug_config' in model_config:
		noise_aug_config = model_config['noise_aug_config']

	model_type = model_base.ModelType.EPS
	if 'parameterization' in model_config:
		if model_config['parameterization'] == "v":
			model_type = model_base.ModelType.V_PREDICTION

	unet = Unet(
			model_name,
			base_type,
			model=model.get_submodule('model.diffusion_model'),
			params=unet_config,
	)
	if fp16:
		unet.half()
	clip = None
	vae = None
	if base_type == 'sd1':
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
					model_name,
					base_type,
					model=model.get_submodule('model.cond_stage_model'),
					params=clip_config,
			)
		if get_vae:
			vae = Vae(
					model_name,
					base_type,
					model=model.get_submodule('model.first_stage_config'),
					params=vae_config,
			)
	else:
		raise ValueError("Unsupported model base type.")
		log.error(f'Failed to get model: {model_name}')
	return LoadedModel(
			model_name,
			base_type,
			scale_factor,
			state_dict=state_dict,
			unet=unet,
			clip=clip,
			vae=vae)

def txt2img(
		self,
		loaded_model,
		prompts,
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
	model = loaded_model.unet.load()
	clip = loaded_model.clip.load()
	vae = loaded_model.vae.load()

	if clip_skip is not None:
		self.clip.skip(clip_skip)

	seed_everything(seed)

	latents = loaded_model.generate_empty_latents((batch_size, height, width))

	if sampler is not None:
		sampler = Sampler(sampler)
	else:
		sampler = Sampler('DDIM')

	samples = []
	with torch.no_grad():
		with model.ema_scope():
			for i in range(batch_size):
				latent = latents[i]
				samples.append(
						sampler.sample(
								model,
								clip,
								latent,
								prompts,
								seed,
								cfg,
								steps,
								width,
								height,
								batch_size
						)
				)

	output = []
	for sample in samples:
		output.append(self.vae.decode(vae, sample))
	return output





