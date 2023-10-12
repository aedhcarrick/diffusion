#  cli_args.py


import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--debug", action="store_true", help="enable debug messages to console")
parser.add_argument("--verbose", action="store_true", help="enable info messages")
parser.add_argument("--log", action="store_true", help="enable debugging to file")


device_group = parser.add_mutually_exclusive_group()
device_group.add_argument("--cpu", action="store_true", help="set prefered device to cpu")
device_group.add_argument("--gpu", action="store_true", help="set prefered device to gpu")
device_group.add_argument("--xpu", action="store_true", help="set prefered device to xpu")
device_group.add_argument("--mps", action="store_true", help="set prefered device to mps")
device_group.add_argument("--directml", action="store_true", help="set prefered device to directml")





args = parser.parse_args()
