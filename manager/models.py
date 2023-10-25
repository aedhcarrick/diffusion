#  manager/models.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


import torch
import utils.devices as _dev


class BaseModel():
	dtype = torch.float32

	def dtype(self):
		if self.dtype == torch.float16:
			return 'float16'
		elif self.dtype == torch.bfloat16:
			return 'bfloat16'
		elif self.dtype == torch.float32:
			return 'float32'
		else:
			return 'Unknown dtype.'

	def load(self):
		self.model.to(self.load_device)
		self.current_device = self.load_device

	def unload(self):
		self.model.to(self.unload_device)
		self.current_device = self.unload_device

	def is_loaded(self):
		return self.current_device == self.load_device


class Unet():
	def __init__(self, unet, params, base_type, load):
		self.model = unet
		self.params = params
		self.base_type = base_type
		self.dtype = _dev.unet_dtype()
		self.offload_device = _dev.unet_offload_device
		self.load_device = _dev.unet_load_device
		self.current_device = self.offload_device
		if load:
			self.load()


class Clip():
	def __init__(self, clip, params, base_type, load):
		self.model = clip
		self.params = params
		self.base_type = base_type
		self.dtype = torch.float32
		self.offload_device = _dev.clip_offload_device
		self.load_device = _dev.clip_load_device
		self.current_device = self.offload_device
		if load:
			self.load()


class Vae():
	def __init__(self, vae, params, base_type, load):
		self.model = vae
		self.config = params
		self.base_type = base_type
		self.dtype = _dev.vae_dtype
		self.offload_device = _dev.vae_offload_device
		self.load_device = _dev.vae_load_device
		self.current_device = self.offload_device
		if load:
			self.load()









