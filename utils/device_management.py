#  utils/device_management.py

#
#	Copied/Adapted from:
#	https://github.com/comfyanonymous/ComfyUI/comfy/model_management.py
#	--> GNU GPLv3
#

import os
import psutil
import torch
import sys

from enum import Enum
from utils.cli_args import args
from utils.log_config import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())


try:
	OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except Exception:
	OOM_EXCEPTION = Exception


class VramState(Enum):
	DISABLED = 'disabled'
	NO_VRAM = 'none'
	LOW_VRAM = 'low'
	NORMAL_VRAM = 'normal'
	HIGH_VRAM = 'high'
	SHARED = 'shared'

class Device(Enum):
	CPU = 'cpu'
	GPU = 'gpu'
	XPU = 'xpu'
	MPS = 'mps'


vram_state = VramState.NORMAL_VRAM
set_vram_to = VramState.NORMAL_VRAM
prefered_device = Device.GPU

total_vram = 0

low_vram_available = True
xpu_available = False

direct_ml_enabled = False
if args.directml is not None:
	import torch_directml
	directml_enabled = True
	device_index = args.directml
	if device_index < 0:
		directml_device = torch_directml.device()
	else:
		directml_device = torch_directml.device(device_index)
	log.info(f'Using directml with device:  {torch_directml.device_name(device_index)}')
	lowvram_available = False # <-- ????

try:
	import intel_extension_for_pytorch as ipex
	if torch.xpu.is_available():
		xpu_available = True
except Exception:
	pass

try:
	if torch.backends.mps.is_available():
		prefered_device = Device.MPS
		import torch.mps
except Exception:
	pass

if args.cpu:
	prefered_device = Device.CPU

def is_intel_xpu():
	global prefered_device
	global xpu_available
	if prefered_device == Device.GPU:
		if xpu_available:
			return True
	return False

def get_torch_device():
	global directml_enabled
	global prefered_device
	if directml_enabled:
		global directml_device
		return directml_device
	elif prefered_device == Device.MPS:
		return torch.device("mps")
	elif prefered_device == Device.CPU:
		return torch.device("cpu")
	else:
		if is_intel_xpu():
			return torch.device("xpu")
		else:
			return torch.device(torch.cuda.current_device())

def get_mem_total(device=None, mem_torch_also=False):
	global directml_enabled
	if device is None:
		device = get_torch_device()
	if hasattr(device, 'type') and (device.type == 'cpu' or device.type == 'mps'):
		mem_total = psutil.virtual_memory().total
		mem_total_torch = mem_total
	else:
		if directml_enabled:
			mem_total = 1024 * 1024 * 1024 #TODO
			mem_total_torch = mem_total
		elif is_intel_xpu():
			stats = torch.xpu.memory_stats(device)
			mem_reserved = stats['reserved_bytes.all.current']
			mem_total = torch.xpu.get_device_properties(device).total_memory
			mem_total_torch = mem_reserved
		else:
			stats = torch.cuda.memory_stats(device)
			mem_reserved = stats['reserved_bytes.all.current']
			_, mem_total_cuda = torch.cuda.mem_get_info(device)
			mem_total_torch = mem_reserved
			mem_total = mem_total_cuda

	if mem_torch_also:
		return (mem_total, mem_total_torch)
	else:
		return mem_total

total_vram = get_mem_total() / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
log.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

if not args.normalvram and not args.cpu:
	if lowvram_available and total_vram <= 4096:
		log.info("Trying to enable lowvram mode because your GPU seems to have <4GB. If you don't want this use: --normalvram")
		set_vram_to = VramState.LOW_VRAM

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

def is_nvidia():
	global prefered_device
	if prefered_device == Device.GPU:
		if torch.version.cuda:
			return True
	return False

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
	ENABLE_PYTORCH_ATTENTION = True
	XFORMERS_IS_AVAILABLE = False

VAE_DTYPE = torch.float32

try:
	if is_nvidia():
		torch_version = torch.version.__version__
		if int(torch_version[0]) >= 2:
			if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
				ENABLE_PYTORCH_ATTENTION = True
			if torch.cuda.is_bf16_supported():
				VAE_DTYPE = torch.bfloat16
	if is_intel_xpu():
		if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
			ENABLE_PYTORCH_ATTENTION = True
except:
	pass

if is_intel_xpu():
	VAE_DTYPE = torch.bfloat16

if args.fp16_vae:
	VAE_DTYPE = torch.float16
elif args.bf16_vae:
	VAE_DTYPE = torch.bfloat16
elif args.fp32_vae:
	VAE_DTYPE = torch.float32


if ENABLE_PYTORCH_ATTENTION:
	torch.backends.cuda.enable_math_sdp(True)
	torch.backends.cuda.enable_flash_sdp(True)
	torch.backends.cuda.enable_mem_efficient_sdp(True)

if args.lowvram:
	set_vram_to = VramState.LOW_VRAM
	lowvram_available = True
elif args.novram:
	set_vram_to = VramState.NO_VRAM
elif args.highvram or args.gpu_only:
	vram_state = VramState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.force_fp32:
	FORCE_FP32 = True

if args.force_fp16:
	FORCE_FP16 = True

if lowvram_available:
	try:
		import accelerate
		if set_vram_to in (VramState.LOW_VRAM, VramState.NO_VRAM):
			vram_state = set_vram_to
	except Exception as e:
		import traceback
		print(traceback.format_exc())
		log.error(f'LOW VRAM MODE NEEDS accelerate.')
		lowvram_available = False

if prefered_device != Device.GPU:
	vram_state = VramState.DISABLED

if prefered_device == Device.MPS:
	vram_state = VramState.SHARED

log.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
	log.info("Disabling smart memory management")

def get_torch_device_name(device):
	if hasattr(device, 'type'):
		if device.type == "cuda":
			try:
				allocator_backend = torch.cuda.get_allocator_backend()
			except:
				allocator_backend = ""
			return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
		else:
			return "{}".format(device.type)
	elif is_intel_xpu():
		return "{} {}".format(device, torch.xpu.get_device_name(device))
	else:
		return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

try:
	log.info(f'Device: {get_torch_device_name(get_torch_device())}')
except:
	log.error('Could not pick default device.')

log.info(f'VAE dtype: {VAE_DTYPE}')

def dtype_size(dtype):
	dtype_size = 4
	if dtype == torch.float16 or dtype == torch.bfloat16:
		dtype_size = 2
	return dtype_size

def unet_offload_device():
	if vram_state == VramState.HIGH_VRAM:
		return get_torch_device()
	else:
		return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
	torch_dev = get_torch_device()
	if vram_state == VramState.HIGH_VRAM:
		return torch_dev
	cpu_dev = torch.device("cpu")
	if DISABLE_SMART_MEMORY:
		return cpu_dev

	model_size = dtype_size(dtype) * parameters

	mem_dev = get_free_memory(torch_dev)
	mem_cpu = get_free_memory(cpu_dev)
	if mem_dev > mem_cpu and model_size < mem_dev:
		return torch_dev
	else:
		return cpu_dev

def unet_dtype(device=None, model_params=0):
	if args.bf16_unet:
		return torch.bfloat16
	if should_use_fp16(device=device, model_params=model_params):
		return torch.float16
	return torch.float32

def text_encoder_offload_device():
	if args.gpu_only:
		return get_torch_device()
	else:
		return torch.device("cpu")

def text_encoder_device():
	if args.gpu_only:
		return get_torch_device()
	elif vram_state == VramState.HIGH_VRAM or vram_state == VramState.NORMAL_VRAM:
		if is_intel_xpu():
			return torch.device("cpu")
		if should_use_fp16(prioritize_performance=False):
			return get_torch_device()
		else:
			return torch.device("cpu")
	else:
		return torch.device("cpu")

def vae_device():
	return get_torch_device()

def vae_offload_device():
	if args.gpu_only:
		return get_torch_device()
	else:
		return torch.device("cpu")

def vae_dtype():
	global VAE_DTYPE
	return VAE_DTYPE

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
		elif is_intel_xpu():
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
	global prefered_device
	if prefered_device != Device.GPU:
		return False
	if is_intel_xpu():
		return False
	if directml_enabled:
		return False
	return XFORMERS_IS_AVAILABLE

def xformers_enabled_vae():
	enabled = xformers_enabled()
	if not enabled:
		return False
	return XFORMERS_ENABLED_VAE

def pytorch_attention_enabled():
	global ENABLE_PYTORCH_ATTENTION
	return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention():
	global ENABLE_PYTORCH_ATTENTION
	if ENABLE_PYTORCH_ATTENTION:
		if is_nvidia():
			return True
	return False

def get_free_memory(dev=None, torch_free_too=False):
	global directml_enabled
	if dev is None:
		dev = get_torch_device()

	if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
		mem_free_total = psutil.virtual_memory().available
		mem_free_torch = mem_free_total
	else:
		if directml_enabled:
			mem_free_total = 1024 * 1024 * 1024 #TODO
			mem_free_torch = mem_free_total
		elif is_intel_xpu():
			stats = torch.xpu.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_allocated = stats['allocated_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = torch.xpu.get_device_properties(dev).total_memory - mem_allocated
		else:
			stats = torch.cuda.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = mem_free_cuda + mem_free_torch

	if torch_free_too:
		return (mem_free_total, mem_free_torch)
	else:
		return mem_free_total

def batch_area_memory(area):
	if xformers_enabled() or pytorch_attention_flash_attention():
		return (area / 20) * (1024 * 1024)
	else:
		return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

def maximum_batch_area():
	global vram_state
	if vram_state == VRAMState.NO_VRAM:
		return 0

	memory_free = get_free_memory() / (1024 * 1024)
	if xformers_enabled() or pytorch_attention_flash_attention():
		area = 20 * memory_free
	else:
		area = ((memory_free - 1024) * 0.9) / (0.6)
	return int(max(area, 0))

def cpu_mode():
	global cpu_state
	return cpu_state == CPUState.CPU

def mps_mode():
	global cpu_state
	return cpu_state == CPUState.MPS

def is_device_cpu(device):
	if hasattr(device, 'type'):
		if (device.type == 'cpu'):
			return True
	return False

def is_device_mps(device):
	if hasattr(device, 'type'):
		if (device.type == 'mps'):
			return True
	return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
	global directml_enabled

	if device is not None:
		if is_device_cpu(device):
			return False

	if FORCE_FP16:
		return True

	if device is not None: #TODO
		if is_device_mps(device):
			return False

	if FORCE_FP32:
		return False

	if directml_enabled:
		return False

	if cpu_mode() or mps_mode():
		return False #TODO ?

	if is_intel_xpu():
		return True

	if torch.cuda.is_bf16_supported():
		return True

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

def soft_empty_cache(force=False):
	global cpu_state
	if cpu_state == CPUState.MPS:
		torch.mps.empty_cache()
	elif is_intel_xpu():
		torch.xpu.empty_cache()
	elif torch.cuda.is_available():
		if force or is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()

def resolve_lowvram_weight(weight, model, key):
	if weight.device == torch.device("meta"): #lowvram NOTE: this depends on the inner working of the accelerate library so it might break.
		key_split = key.split('.')              # I have no idea why they don't just leave the weight there instead of using the meta device.
		op = comfy.utils.get_attr(model, '.'.join(key_split[:-1]))
		weight = op._hf_hook.weights_map[key_split[-1]]
	return weight



