#  manager/models.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


import torch
import utils.devices as _dev
from manager.components import Conditioning, Latents
from manager.components import getTokenizer


class BaseModel():
	dtype = torch.float32

	def __init__(self, name: str):
		self.name = name

	def dtype(self):
		if self.dtype == torch.float16:
			return 'float16'
		elif self.dtype == torch.bfloat16:
			return 'bfloat16'
		elif self.dtype == torch.float32:
			return 'float32'
		else:
			return 'Unknown dtype.'

	def half(self):
		self.model.half()

	def to(self, *args, **kwargs):
		self.model.to(*args, **kwargs)

	def load(self):
		self.model.to(self.load_device)
		self.current_device = self.load_device

	def unload(self):
		self.model.to(self.unload_device)
		self.current_device = self.unload_device

	def is_loaded(self):
		return self.current_device == self.load_device


class Unet(BaseModel):
	def __init__(self, name=None, model=None, params=None, base_type='sd1', load=False):
		super().__init__(name)
		self.model = unet
		self.params = params
		self.base_type = base_type
		self.dtype = _dev.unet_dtype()
		self.offload_device = _dev.unet_offload_device
		self.load_device = _dev.unet_load_device
		self.current_device = self.offload_device
		if load:
			self.load()

	def eval(self):
		self.model.eval()


class Clip(BaseModel):
	def __init__(self, name=None, model=None, params=None, base_type='sd1', load=False):
		super().__init__(name)
		self.model = model
		self.tokenizer = getTokenizer(base_type)
		self.params = params
		self.base_type = base_type
		self.dtype = torch.float32
		self.offload_device = _dev.clip_offload_device
		self.load_device = _dev.clip_load_device
		self.current_device = self.offload_device
		if load:
			self.load()

	def encode(self, text: str) -> Conditioning:
		tokens = self.tokenizer.tokenize_with_weights(text)
		return self.encode_from_tokens(tokens)

	def encode_from_tokens(self, tokens) -> Conditioning:
		if not self.is_loaded():
			self.load()
		return self.model.encode_token_weights(tokens)


class Vae(BaseModel):
	def __init__(self, name=None, model=None, params=None, base_type='sd1', load=False):
		super().__init__(name)
		self.model = vae
		self.config = params
		self.base_type = base_type
		self.dtype = _dev.vae_dtype
		self.offload_device = _dev.vae_offload_device
		self.load_device = _dev.vae_load_device
		self.current_device = self.offload_device
		if load:
			self.load()

	def decode(self, samples):
		if not self.is_loaded():
			self.load()
		x_samples = self.model.decode_first_stage(samples)
		x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
		x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
		x_samples_torch = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
		images = []
		for x_sample in x_samples_torch:
			x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
			img = Image.fromarray(x_sample.astype(np.uint8))
			images.append(img)
		return images



