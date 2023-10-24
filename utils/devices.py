#  utils/devices.py


#
#	adapted from https://github.com/comfyanonymous/confyui
#	-->GPLv3
#


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())
log.info(f'Initializing devices..')


from enum import Enum


class VRAMSTATE(Enum):
	NONE = 0
	LOW = 1
	NORMAL = 2
	HIGH = 3
	SHARED = 4


class DEVICE(Enum):
	CPU = 0
	GPU = 1
	XPU = 2
	MPS = 3


import psutil
import torch


use_device = DEVICE.GPU
vram_state = VRAMSTATE.NORMAL
low_vram_ok = True
directml_enabled = False
xpu_available = False
torch_device = None


def get_device_name(device):
	if hasattr(device, 'type'):
		if device.type == "cuda":
			try:
				allocator_backend = torch.cuda.get_allocator_backend()
			except:
				allocator_backend = ""
			return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
		else:
			return "{}".format(device.type)
	elif use_device == DEVICE.XPU:
		return "{} {}".format(device, torch.xpu.get_device_name(device))
	else:
		return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

#	get torch device

from utils.args import args

if args.directml is not None:
	import torch_directml
	directml_enabled = True
	index = args.directml
	if index < 0:
		torch_device = torch_directml.device()
	else:
		torch_device = torch_directml.device(index)
		log.info(f'Directml enabled: {torch_directml.device_name(index)}')
		low_vram_ok = False
else:
	try:
		import intel_extension_for_pytorch as ipex
		if torch.xpu.is_available():
			use_device = DEVICE.XPU
	except Exception:
		pass

	try:
		if torch.backends.mps.is_available():
			import torch.mps
			use_device = DEVICE.MPS
	except Exception:
		pass

if args.novram:
	set_vram_to = VRAMSTATE.NONE
elif args.lowvram:
	set_vram_to = VRAMSTATE.LOW
elif args.highvram:
	set_vram_to = VRAMSTATE.HIGH
	low_vram_ok = False
else:
	set_vram_to = VRAMSTATE.NORMAL

if args.cpu:
	use_device = DEVICE.CPU
	set_vram_to = VRAMSTATE.NONE

if not directml_enabled:
	if use_device == DEVICE.MPS:
		torch_device = torch.device('mps')
	elif use_device == DEVICE.XPU:
		torch_device = torch.device('xpu')
	elif use_device == DEVICE.CPU:
		torch_device = torch.device('cpu')
		low_vram_ok = False
	else:
		if torch.cuda.is_available():
			torch_device = torch.device(torch.cuda.current_device())
		else:
			use_device = DEVICE.CPU
			torch_device = torch.device('cpu')
	log.info(f'Using torch device:  {get_device_name(torch_device)}')

#	get total memory resources

def get_total_memory(device=None):
	if device == None:
		device = torch_device
	if directml_enabled:
		total = pow(1024, 3)
	else:
		if (use_device == DEVICE.CPU or use_device == DEVICE.MPS):
			total = psutil.virtual_memory().total
		elif use_device == DEVICE.XPU:
			total = torch.xpu.get_device_properties(device).total_memory
		else:
			_, total = torch.cuda.mem_get_info(device)
	return total

def get_free_memory(device=None):
	if device == None:
		device = torch_device
	if directml_enabled:
		free = pow(1024, 3)
	else:
		if (use_device == DEVICE.CPU or use_device == DEVICE.MPS):
			free = psutil.virtual_memory().available
		elif use_device == DEVICE.XPU:
			stats = torch.xpu.memory_stats(device)
			allocated = stats['allocated_bytes.all.current']
			total = torch.xpu.get_device_properties(device).total_memory
			free = total - allocated
		else:
			stats = torch.cuda.memory_stats(device)
			active = stats['active_bytes.all.current']
			reserved = stats['reserved_bytes.all.current']
			cuda, _ = torch.cuda.mem_get_info(device)
			free = cuda + reserved - active
	return free

total_ram = psutil.virtual_memory().total / (1024 * 1024)
total_vram = get_total_memory() / (1024 * 1024)

log.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

#	set vram state

if not args.normalvram and use_device is not DEVICE.CPU:
	if low_vram_ok and total_vram <= 4096:
		log.info('Enabling low vram mode..')
		set_vram_to = VRAMSTATE.LOW

if low_vram_ok:
	try:
		import accelerate
		if set_vram_to in (VRAMSTATE.LOW, VRAMSTATE.NONE):
			vram_state = set_vram_to
	except Exception as e:
		import traceback
		print(traceback.format_exc())
		log.error(f'LOW VRAM MODE NEEDS accelerate.')
		low_vram_ok = False

if use_device == DEVICE.MPS:
	vram_state = VRAMSTATE.SHARED
elif use_device != DEVICE.GPU:
	vram_state = VRAMSTATE.NONE

log.info(f"Set vram state to: {vram_state.name}")

#	check for xformers and pytorch attention

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
	XFORMERS_IS_AVAILABLE = False
else:
	try:
		import xformers
		import xformers.ops
		XFORMERS_IS_AVAILABLE = True
		try:
			XFORMERS_VERSION = xformers.version.__version__
			log.info(f'xformers version: {XFORMERS_VERSION}')
			if XFORMERS_VERSION.startswith("0.0.18"):
				log.warning("This version of xformers has a major bug where you will get black images when generating high resolution images.")
				log.warning("Please downgrade or upgrade xformers to a different version.")
				XFORMERS_ENABLED_VAE = False
		except:
			pass
	except:
		XFORMERS_IS_AVAILABLE = False

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
	ENABLE_PYTORCH_ATTENTION = True
	XFORMERS_IS_AVAILABLE = False

VAE_DTYPE = torch.float32

try:
	if (use_device == DEVICE.GPU and torch.version.cuda):
		torch_version = torch.version.__version__
		if int(torch_version[0]) >= 2:
			if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
				ENABLE_PYTORCH_ATTENTION = True
			if torch.cuda.is_bf16_supported():
				VAE_DTYPE = torch.bfloat16
	if use_device == DEVICE.XPU:
		if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
			ENABLE_PYTORCH_ATTENTION = True
except:
	pass

if ENABLE_PYTORCH_ATTENTION:
	torch.backends.cuda.enable_math_sdp(True)
	torch.backends.cuda.enable_flash_sdp(True)
	torch.backends.cuda.enable_mem_efficient_sdp(True)

#	set vae dtype

FORCE_FP32 = False
FORCE_FP16 = False

if args.force_fp32:
	FORCE_FP32 = True
elif args.force_fp16:
	FORCE_FP16 = True

if use_device == DEVICE.XPU:
	VAE_DTYPE = torch.bfloat16

log.info(f'VAE dtype: {VAE_DTYPE}')

# set model load and offload targets

if vram_state == VRAMSTATE.HIGH:
	unet_offload_device = torch_device
	unet_load_device = torch_device
	text_encoder_offload_device = torch_device
	text_encoder_device = torch_device
	vae_offload_device = torch_device
else:
	unet_offload_device = torch.device('cpu')
	unet_load_device = torch.device('cpu')
	text_encoder_offload_device = torch.device('cpu')
	text_encoder_device = torch.device('cpu')
	vae_offload_device = torch.device('cpu')

vae_load_device = torch_device

#	helper functions

def unet_dtype(device=None, model_params=0):
	if should_use_fp16(device=device, model_params=model_params):
		return torch.float16
	return torch.float32

def get_autocast_device(dev):
	if hasattr(dev, 'type'):
		return dev.type
	return "cuda"

def cast_to_device(tensor, device, dtype, copy=False):
	device_supports_cast = False
	if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
		device_supports_cast = True
	elif tensor.dtype == torch.bfloat16:
		if hasattr(device, 'type') and device.type.startswith("cuda"):
			device_supports_cast = True
		elif use_device == DEVICE.XPU:
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

def xformers_enabled():
	global directml_enabled
	global use_device
	if use_device != DEVICE.GPU:
		return False
	if use_device == DEVICE.XPU:
		return False
	if directml_enabled:
		return False
	return XFORMERS_IS_AVAILABLE

def xformers_enabled_vae():
	if not xformers_enabled():
		return False
	return XFORMERS_ENABLED_VAE

def pytorch_attention_enabled():
	global ENABLE_PYTORCH_ATTENTION
	return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention():
	global ENABLE_PYTORCH_ATTENTION
	if ENABLE_PYTORCH_ATTENTION:
		if (use_device == DEVICE.GPU and torch.version.cuda):
			return True
	return False

def batch_area_memory(area):
	if xformers_enabled() or pytorch_attention_flash_attention():
		return (area / 20) * (1024 * 1024)
	else:
		return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

def maximum_batch_area():
	if vram_state == VRAMSTATE.NONE:
		return 0
	memory_free = get_free_memory() / (1024 * 1024)
	if xformers_enabled() or pytorch_attention_flash_attention():
		area = 20 * memory_free
	else:
		area = ((memory_free - 1024) * 0.9) / (0.6)
	return int(max(area, 0))

def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
	global directml_enabled
	if (directml_enabled or use_device == DEVICE.CPU or use_device == DEVICE.MPS):
		return False
	if (use_device == DEVICE.XPU or torch.cuda.is_bf16_supported()):
		return True

	if FORCE_FP16:
		return True

	if FORCE_FP32:
		return False

	props = torch.cuda.get_device_properties("cuda")
	if props.major < 6:
		return False

	fp16_works = False
	#FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
	#when the model doesn't actually fit on the card
	#TODO: actually test if GP106 and others have the same type of behavior
	nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050"]
	for x in nvidia_10_series:
		if x in props.name.lower():
			fp16_works = True

	if fp16_works:
		free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
		if (not prioritize_performance) or model_params * 4 > free_model_memory:
			return True

	if props.major < 7:
		return False

	#FP16 is just broken on these cards
	nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
	for x in nvidia_16_series:
		if x in props.name:
			return False

	return True

def empty_cache():
	if use_device == DEVICE.MPS:
		torch.mps.empty_cache()
	elif use_device == DEVICE.XPU:
		torch.xpu.empty_cache()
	elif (use_device == DEVICE.GPU and torch.version.cuda):
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()

'''
def resolve_lowvram_weight(weight, model, key):
	if weight.device == torch.device("meta"): #lowvram NOTE: this depends on the inner working of the accelerate library so it might break.
		key_split = key.split('.')              # I have no idea why they don't just leave the weight there instead of using the meta device.
		op = comfy.utils.get_attr(model, '.'.join(key_split[:-1]))
		weight = op._hf_hook.weights_map[key_split[-1]]
	return weight
'''





