#  models.py


import copy
import inspect
import torch

from enum import Enum
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
from ldm.modules.diffusionmodules.util import make_beta_schedule
from utils.log_config import ThreadContextFilter
from uitls import device_manager as _utils

from clip import Clip
from vae import Vae


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


def get_state_dict(ckpt):
	with open(ckpt,'rb') as f:
		if ckpt.lower().endswith() == '.safetensors':
			log.info(f'Getting state dict from safetensors..')
			buffer = f.read()
			state_dict = safetensors.torch.load(buffer)
		else:
			log.info(f'Getting state dict from checkpoint..')
			buffer = io.BytesIO(f.read())
			pl_sd = torch.load(buffer, map_location=torch.device('cpu'))
			state_dict = pl_sd["state_dict"]
	return state_dict

def load_model_weights(model, state_dict):
	m, u = model.load_state_dict(state_dict, strict=False)
	if len(m) > 0:
		log.error(f"missing keys:  {m}")
	if len(u) > 0:
		log.error(f"unexpected keys:  {u}")
	if not len(m) + len(u):
		log.info('All keys matched.')
	return model

def load_clip_weights(model, state_dict):
	k = list(state_dict.keys())
	for x in k:
		if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
			y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
			state_dict[y] = state_dict.pop(x)

	if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
		ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
		if ids.dtype == torch.float32:
			state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

	prefix_from = "cond_stage_model.model."
	prefix_to = "cond_stage_model.transformer.text_model."
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in state_dict:
            state_dict[keys_to_replace[k].format(prefix_to)] = state_dict.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }
	num_of_blocks = 24
    for resblock in range(number_of_blocks):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in state_dict:
                    state_dict[k_to] = state_dict.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in state_dict:
                weights = state_dict.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    state_dict[k_to] = weights[shape_from*x:shape_from*(x + 1)]
	return load_model_weights(model, state_dict)

def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, 'type') and device.type.startswith("cuda"):
            device_supports_cast = True
        elif _utils.is_xpu():
            device_supports_cast = True

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy)
            return tensor.to(device, copy=copy).to(dtype)
        else:
            return tensor.to(device).to(dtype)
    else:
        return tensor.to(dtype).to(device, copy=copy)


class ModelType(Enum):
	EPS = 'eps'
	V_PREDICTION = 'v_prediction'


class ModelConfig():
	unet_config = {}
	vae_config = {}
	clip_config = {}
	clip_prefix = []
	clip_vision_prefix = None
	noise_aug_config = None
	beta_schedule = "linear"
	latent_format = LatentFormatSD1_5(scale_factor)
	model_type = ModelType.EPS
	dtype = torch.float32

	def __init__(self, model_name, model_path):
		self.name = model_name
		self.path = model_path
		self.state_dict = get_state_dict(model_path)


	def matches(s, unet_config):
		for k in s.unet_config:
			if s.unet_config[k] != unet_config[k]:
				return False
		return True

	def is_inpaint(self):
		return self.unet_config["in_channels"] > 4


class BaseModel(torch.nn.Module):
	def __init__(self, model_config, device):
		super().__init__()
		self.model_config = model_config
		self.latent_format = model_config.latent_format
		self.register_schedule(
				given_betas=None,
				beta_schedule=model_config.beta_schedule,
				timesteps=1000,
				linear_start=0.00085,
				linear_end=0.012,
				cosine_s=8e-3
		)
		unet_config = model_config.unet_config
		if not unet_config.get("disable_unet_model_creation", False):
			self.diffusion_model = UNetModel(**unet_config, device=device)
		self.model_type = model_config.model_type
		self.adm_channels = unet_config.get("adm_in_channels", None)
		if self.adm_channels is None:
			self.adm_channels = 0

	def register_schedule(
			self,
			given_betas=None,
			beta_schedule="linear",
			timesteps=1000,
			linear_start=1e-4,
			linear_end=2e-2,
			cosine_s=8e-3
			):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

    def apply_model(
			self,
			x,
			t,
			c_concat=None,
			c_crossattn=None,
			c_adm=None,
			control=None,
			transformer_options={}
			):
        if c_concat is not None:
            xc = torch.cat([x] + [c_concat], dim=1)
        else:
            xc = x
        context = c_crossattn
        dtype = self.get_dtype()
        xc = xc.to(dtype)
        t = t.to(dtype)
        context = context.to(dtype)
        if c_adm is not None:
            c_adm = c_adm.to(dtype)
        return self.diffusion_model(xc, t, context=context, y=c_adm, control=control, transformer_options=transformer_options).float()

	def get_dtype(self):
		return self.diffusion_model.dtype

	def is_adm(self):
		return self.adm_channels > 0

    def load_model_weights(self):
		to_load = {}
		state_dict = self.model_config.state_dict
		unet_prefix = "model.diffusion_model."
		for key in list(state_dict.keys()):
			if key.startswith(unet_prefix):
				to_load[key[len(unet_prefix):]] = state_dict.pop(k)
		m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
		if len(m) > 0:
			log.error(f"missing keys:  {m}")
		if len(u) > 0:
			log.error(f"unexpected keys:  {u}")
		if not len(m) + len(u):
			log.info('All keys matched.')
		return self

	def process_latent_in(self, latent):
		return self.latent_format.process_in(latent)

	def process_latent_out(self, latent):
		return self.latent_format.process_out(latent)

    def set_inpaint(self):
        self.concat_keys = ("mask", "masked_image")


class SD2_1_UNCLIP(BaseModel):
	def __init__(self, model_config, device=None):
		super().__init__(model_config, device=device)
		noise_aug_config = model_config.noise_aug_config
		self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

	def encode_adm(self, **kwargs):
		unclip_conditioning = kwargs.get("unclip_conditioning", None)
		device = kwargs["device"]
		if unclip_conditioning is None:
			return torch.zeros((1, self.adm_channels))
		else:
			return unclip_adm(unclip_conditioning, device, self.noise_augmentor, kwargs.get("unclip_noise_augment_merge", 0.05))


class LoadedModel():
	def __init__(self, model, load_device, offload_device):
		self.config = model.model_config
		self.name = model.model_config.name
		self.size = 0

		log.info(f'Loading model:  {self.name}..')
		self.model = model
		self.accelerated = False
		self.patches = {}
		self.backup = {}
		self.options = {
			"transformer_options": {}
		}
		self.clip = None
		self.vae = None
		self.model_size()

		self.load_device = load_device
		self.offload_device = offload_device
		self.model = model.to(offload_device)
		self.model.load_model_weights()
		self.current_device = self.offload_device
		log.info(f'Model loaded to {self.offload_device}')

	def model_size(self):
		state_dict = self.model.state_dict
		if self.size == 0:
			for k in state_dict:
				t = state_dict[k]
				self.size += t.nelement() * t.element_size()
			self.model_keys = set(state_dict.keys())
		return self.size

	def memory_required(self, device):
		if device == self.current_device:
			return 0
		else:
			return self.model_size

	def get_dtype(self):
		if hasattr(self.model, 'get_dtype'):
			returnn self.model.get_dtype()

	def __eq__(self, other):
		return self.model is other.model

	def get_clip(self):
		log.info(f'Loading CLIP from model..')
		weights = torch.nn.Module()
		clip = Clip(self.config.clip_config)
		weights.cond_stage_model = clip.cond_stage_model
		load_clip_weights(weights, self.state_dict)
		return clip

	def get_vae(self):
		log.info(f'Loading VAE from model..')
		weights = torch.nn.Module()
		vae = Vae(self.config.vae_config)
		weights.first_stage_model = vae.first_stage_model
		load_model_weights(weights, state_dict)
		return vae

	def load_model(self, low_vram_mem=0):
		patch_to = None
		if low_vram_mem == 0:
			patch_to = self.load_device
		self.model_patches_to(self.load_device)
		self.model_patches_to(self.get_dtype())

		try:
			self.real_model = self.patch_model(device=patch_to)
		except Exception as e:
			self.unpatch_model(self.offload_device)
			self.unload_model()
			log.error(f'Failed to load model: {e}')

		if low_vram_mem > 0:
			log.info(f'Loading model in low vram mode')
			device_map = accelerate.infer_auto_device_map(
					self.real_model,
					max_memory={
						0: "{}MiB".format(lowvram_model_memory // (1024 * 1024)),
						"cpu": "16GiB"
					}
			)
			accelerate.dispatch_model(
					self.real_model,
					device_map=device_map,
					main_device=self.load_device
			)
			self.accelerated = True
		if _utils.is_xpu and not args.disable_ipex:
			self.real_model = torch.xpu.optimize(
					self.real_model.eval(),
					inplace=True,
					auto_kernel_selection=True,
					graph_mode=True,
			)
		return self.real_model

	def unload_model(self):
		if self.accelerated:
			accelerate.hooks.remove_hook_from_submodules(self.real_model)
			self.accelerated = False
		self.unpatch_model(self.offload_device)
		self.model_patches_to(self.offload_device)

	def clone(self):
		n = LoadedModel(self.model, self.load_device, self.offload_device)
		n.patches = {}
		for k in self.patches:
			n.patches[k] = self.patches[k][:]
		n.model_options = copy.deepcopy(self.model_options)
		n.model_keys = self.model_keys
		return n

	def is_clone(self, other):
		if hasattr(other, 'model') and self.model is other.model:
			return True
		return False

	def set_model_sampler_cfg_function(self, sampler_cfg_function):
		if len(inspect.signature(sampler_cfg_function).parameters) == 3:
			self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
		else:
			self.model_options["sampler_cfg_function"] = sampler_cfg_function

	def set_model_unet_function_wrapper(self, unet_wrapper_function):
		self.model_options["model_function_wrapper"] = unet_wrapper_function

	def set_model_patch(self, patch, name):
		if "patches" not in self.model_options["transformer_options"]:
			self.model_options["transformer_options"]["patches"] = {}
		self.model_options["transformer_options"]["patches"][name] = self.model_options["transformer_options"]["patches"].get(name, []) + [patch]

	def set_model_patch_replace(self, patch, name, block_name, number):
		if "patches_replace" not in self.model_options["transformer_options"]:
			self.model_options["transformer_options"]["patches_replace"] = {}
		if name not in self.model_options["transformer_options"]["patches_replace"]:
			self.model_options["transformer_options"]["patches_replace"][name] = {}
		self.model_options["transformer_options"]["patches_replace"][name][(block_name, number)] = patch

	def model_patches_to(self, device):
		if "patches" in self.model_options["transformer_options"]:
			patches = self.model_options["transformer_options"]["patches"]
			for name in patches:
				patch_list = patches[name]
				for i in range(len(patch_list)):
					if hasattr(patch_list[i], "to"):
						patch_list[i] = patch_list[i].to(device)
		if "patches_replace" in self.model_options["transformer_options"]:
			patches = self.model_options["transformer_options"]["patches_replace"]
			for name in patches:
				patch_list = patches[name]
				for k in patch_list:
					if hasattr(patch_list[k], "to"):
						patch_list[k] = patch_list[k].to(device)
		if "unet_wrapper_function" in self.model_options:
			wrap_func = self.model_options["unet_wrapper_function"]
			if hasattr(wrap_func, "to"):
				self.model_options["unet_wrapper_function"] = wrap_func.to(device)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

	def patch_model(self, device=None):
		state_dict = self.model_config.state_dict
		for key in self.patches:
			if key not in state_dict:
				log.info(f"Could not patch. Key doesn't exist in model: {key}")
				continue
			weight = state_dict[key]
			if key not in self.backup:
				self.backup[key] = weight.to(self.offload_device)
			if device is not None:
				temp_weight = cast_to_device(weight, device, torch.float32, copy=True)
			else:
				temp_weight = weight.to(torch.float32, copy=True)
			out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
			self.set_attr(key, out_weight)

		if device is not None:
			self.model.to(device)
			self.current_device = device
		return self.model

	def unpatch_model(self, device=None):
		keys = list(self.backup.keys())
		for key in keys:
			self.set_attr(key, self.backup[key])
		self.backup = {}
		if device is not None:
			self.model.to(device)
			self.current_device = device

	def set_attr(self, attr, value):
		attrs = attr.split(".")
		obj = self.model
		for name in attrs[:-1]:
			obj = getattr(obj, name)
		prev = getattr(obj, attrs[-1])
		setattr(obj, attrs[-1], torch.nn.Parameter(value))
		del prev

	def get_attr(self, attr):
		attrs = attr.split(".")
		obj = self.model
		for name in attrs:
			obj = getattr(obj, name)
		return obj

	def calculate_weight(self, patches, weight, key):
		for p in patches:
			alpha = p[0]
			v = p[1]
			strength_model = p[2]
			if strength_model != 1.0:
				weight *= strength_model

			if isinstance(v, list):
				v = (self.calculate_weight(v[1:], v[0].clone(), key), )

			if len(v) == 1:
				w1 = v[0]
				if alpha != 0.0:
					if w1.shape != weight.shape:
						log.warning(f"Shape mismatch:  {key}. Weights not merged {w1.shape} != {weight.shape}")
					else:
						weight += alpha * cast_to_device(w1, weight.device, weight.dtype)

			elif len(v) == 4: #lora/locon
				mat1 = cast_to_device(v[0], weight.device, torch.float32)
				mat2 = cast_to_device(v[1], weight.device, torch.float32)
				if v[2] is not None:
					alpha *= v[2] / mat2.shape[0]
				if v[3] is not None:
					# locon weights <-- double check this
					mat3 = cast_to_device(v[3], weight.device, torch.float32)
					final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
					mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
				try:
					weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
				except Exception as e:
					log.error(f'Key: {key}  : {e}')

			elif len(v) == 8: #lokr
				w1 = v[0]
				w2 = v[1]
				w1_a = v[3]
				w1_b = v[4]
				w2_a = v[5]
				w2_b = v[6]
				t2 = v[7]
				dim = None

				if w1 is None:
					dim = w1_b.shape[0]
					w1 = torch.mm(
							cast_to_device(w1_a, weight.device, torch.float32),
							cast_to_device(w1_b, weight.device, torch.float32)
					)
				else:
					w1 = cast_to_device(w1, weight.device, torch.float32)

				if w2 is None:
					dim = w2_b.shape[0]
					if t2 is None:
						w2 = torch.mm(
								cast_to_device(w2_a, weight.device, torch.float32),
								cast_to_device(w2_b, weight.device, torch.float32)
						)
					else:
						w2 = torch.einsum(
								'i j k l, j r, i p -> p r k l',
								cast_to_device(t2, weight.device, torch.float32),
								cast_to_device(w2_b, weight.device, torch.float32),
								cast_to_device(w2_a, weight.device, torch.float32)
						)
				else:
					w2 = cast_to_device(w2, weight.device, torch.float32)

				if len(w2.shape) == 4:
					w1 = w1.unsqueeze(2).unsqueeze(2)
				if v[2] is not None and dim is not None:
					alpha *= v[2] / dim

				try:
					weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
				except Exception as e:
					log.error(f'Key: {key}  : {e}')

			else: #loha
				w1a = v[0]
				w1b = v[1]
				if v[2] is not None:
					alpha *= v[2] / w1b.shape[0]
				w2a = v[3]
				w2b = v[4]
				if v[5] is not None: #cp decomposition
					t1 = v[5]
					t2 = v[6]
					m1 = torch.einsum(
							'i j k l, j r, i p -> p r k l',
							cast_to_device(t1, weight.device, torch.float32),
							cast_to_device(w1b, weight.device, torch.float32),
							cast_to_device(w1a, weight.device, torch.float32)
					)
					m2 = torch.einsum(
							'i j k l, j r, i p -> p r k l',
							cast_to_device(t2, weight.device, torch.float32),
							cast_to_device(w2b, weight.device, torch.float32),
							cast_to_device(w2a, weight.device, torch.float32)
					)
				else:
					m1 = torch.mm(
							cast_to_device(w1a, weight.device, torch.float32),
							cast_to_device(w1b, weight.device, torch.float32)
					)
					m2 = torch.mm(
							cast_to_device(w2a, weight.device, torch.float32),
							cast_to_device(w2b, weight.device, torch.float32)
					)
				try:
					weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
				except Exception as e:
					log.error(f'Key: {key}  : {e}')

		return weight







