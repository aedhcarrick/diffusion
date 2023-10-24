#  utils/model_utils.py


import logging
from utils.devices import unet_dtype
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


from paths import get_full_path


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

def load_module_from_config(self, config, ckpt):
		sd = get_state_dict(ckpt)
		model = instantiate_from_config(config.model)
		m, u = model.load_state_dict(sd, strict=False)
		if len(m) > 0:
			self.log.error(f"missing keys:  {m}")
		if len(u) > 0:
			self.log.error(f"unexpected keys:  {u}")
		if len(m) + len(u) == 0:
			self.info(f'All keys matched.')
		return model

def get_model(model_name: str, config: Union[None, str] = None):
	full_path = get_full_path('checkpoints', model_name)
	if not full_path:
		log.info(f'Failed to find model: {model_name}')
		return None
	state_dict = get_state_dict(full_path)
	if config is None:
		config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
	module = load_module_from_config(full_path, config)
	unet = Unet(module.get_sub_module('model'))
	clip = Clip(module.get_submodule('cond_stage_model'))
	vae = Vae(module.get_sub_module('first_stage_model'))

	return unet, clip, vae





