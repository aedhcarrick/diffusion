#  utils/device_management.py

#
#	Adapted from:
#	https://github.com/comfyanonymous/ComfyUI/comfy/model_management.py
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

XFORMERS_VERSION = ""
XFORMERS_ENABLED = True


class VramState(Enum):
	DISABLED = 'disabled'
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
        set_vram_to = VRAMState.LOW_VRAM

###### edit everything below this line

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
            print("xformers version:", XFORMERS_VERSION)
            if XFORMERS_VERSION.startswith("0.0.18"):
                print()
                print("WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                print("Please downgrade or upgrade xformers to a different version.")
                print()
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
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
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.force_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    try:
        import accelerate
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("ERROR: LOW VRAM MODE NEEDS accelerate.")
        lowvram_available = False


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    print("Disabling smart memory management")

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
    print("Device:", get_torch_device_name(get_torch_device()))
except:
    print("Could not pick default device.")

print("VAE dtype:", VAE_DTYPE)

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
