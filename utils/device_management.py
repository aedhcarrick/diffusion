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






def get_torch_device():
	return torch.device(torch.cuda.current_device())

