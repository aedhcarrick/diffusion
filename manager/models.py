#  manager/models.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


import torch
import utils.devices as _dev
from manager.components import Conditioning, Latents
from manager.components import getTextEncoder, getTokenizer, get


class BaseModel():
	dtype = torch.float32
	offload_device = None
	load_device = None
	current_device = None

	def __init__(self, name: str, base_type:str):
		self.name = name
		self.base_type = base_type

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

	def is_loaded(self):
		return self.current_device == self.load_device

	def load(self):
		if not self.is_loaded():
			self.model.to(self.load_device)
			self.current_device = self.load_device
		return self

	def unload(self):
		if self.is_loaded():
			self.model.to(self.unload_device)
			self.current_device = self.unload_device
		return self


class Unet(BaseModel):
	def __init__(self, name, base_type, model=None, params=None, load=False):
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
		else:
			self.unload()

	def eval(self):
		self.model.eval()


class Clip(BaseModel):
	def __init__(self, name, base_type, model=None, params=None, load=False):
		super().__init__(name, base_type)
		if model is not None:
			self.model = model
		else:
			self.model = getTextEncoder(base_type)
		self.tokenizer = getTokenizer(base_type)
		self.params = params
		self.dtype = torch.float32
		self.offload_device = _dev.clip_offload_device
		self.load_device = _dev.clip_load_device
		self.current_device = self.offload_device
		if load:
			self.load()
		else:
			self.unload()

	def encode(self, text: str):
		tokens = self.tokenizer.tokenize_with_weights(text)
		return self.encode_from_tokens(tokens)

	def encode_from_tokens(self, tokens):
		if not self.is_loaded():
			self.load()
		return self.model.encode_token_weights(tokens)

	def get_learned_conditioning(self, cond):
		return self.model.get_learned_conditioning(cond)


class Vae(BaseModel):
	def __init__(self, name, base_type, model=None, params=None, load=False):
		super().__init__(name, base_type)
		if model is not None:
			self.model = model
		else:
			self.model = getVae(base_type, params)
		self.params = params
		self.dtype = _dev.vae_dtype
		self.offload_device = _dev.vae_offload_device
		self.load_device = _dev.vae_load_device
		self.current_device = self.offload_device
		self.scale_factor = 2 ** (len(self.config['block_out_channels']) - 1)
		if load:
			self.load()
		else:
			self.unload()

	def decode(self, samples):
		if not self.is_loaded():
			self.load()
		x_samples = self.model.decode(samples, return_dict=False)
		x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
		x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
		x_samples_torch = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
		images = []
		for x_sample in x_samples_torch:
			x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
			img = Image.fromarray(x_sample.astype(np.uint8))
			images.append(img)
		return images

	def enable_slicing(self):
		self.model.enable_slicing()

	def disable_slicing(self):
		self.model.disable_slicing()

	def enable_tiling(self):
		self.model.enable_tiling()

	def disable_tiling(self):
		self.model disable_tiling()


class LoadedModel():
	def __init__(
			self,
			name,
			base_type,
			scale_factor,
			state_dict=None,
			unet=None,
			clip=None,
			vae=None,
			):
		self.name = name
		self.base_type = base_type
		self.params = state_dict['model']['params']
		self.unet = unet
		self.clip = clip
		self.vae = vae
		self.latents = Latent(self.base_type, scale_factor)
		if self.clip is None:
			if base_type == 'sd1':
				params = self.params['cond_state_model']
			self.clip = Clip(name, base_type, params=params)

	def generate_empty_latents(self, batch_size, height, width):
		return self.latents.generate(
				batch_size,
				height,
				width,
				)



