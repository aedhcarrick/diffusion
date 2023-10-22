#  utils/cli_args.py


import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--debug", action="store_true", help="enable debug messages to console")
parser.add_argument("--verbose", action="store_true", help="enable info messages")
parser.add_argument("--log", action="store_true", help="enable debugging to file")

parser.add_argument("--directml", type="int", metavar="DIRECTML_DEVICE", const=-1, help="set prefered device to directml")
parser.add_argument("--disable-ipex-optimize", action="store_true", help="Disables ipex.optimize when loading models with Intel GPUs.")
parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")
parser.add_argument("--disable-smart-memory", action="store_true", help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")

device_group = parser.add_mutually_exclusive_group()
device_group.add_argument("--cpu", action="store_true", help="set prefered device to cpu")
device_group.add_argument("--gpu", action="store_true", help="set prefered device to gpu")
device_group.add_argument("--xpu", action="store_true", help="set prefered device to xpu")
device_group.add_argument("--mps", action="store_true", help="set prefered device to mps")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")


args = parser.parse_args()
