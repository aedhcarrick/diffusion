#  utils/parse_config.py


default_config = {
	"gpu_only": False,
	"high_vram": False,
	"normal_vram": True,
	"low_vram": False,
	"cpu_only": False,
	"use_directml": False,
	"directml_device": -1,
	"force-fp32": False,
	"force-fp16": False,
	"fp16-vae": False,
	"fp32-vae": False,
	"bf16-vae": False,
	"use_split_cross_attention": False,
	"use_quad_cross_attention": False,
	"use_pytorch_cross_attention": False,
	"disable_xformers": False,
}

import os
from omegaconf import OmegaConf


path_to_config = './model_manager.config'
config = {}

if os.path.exists(path_to_config):
	config = OmegaConf.load(path_to_config)
else:
	config = default_config

