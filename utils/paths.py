#  utils/paths.py


import logging
from utils.logger import ThreadContextFilter


log = logging.getLogger(__name__)
log.addFilter(ThreadContextFilter())

default_paths = {
	'checkpoints': 'models/checkpoints',
	'input_dir': 'input',
	'output_dir': 'output',
}



import os


def get_pathk(path_name: str):
	return default_paths[path_name]

def get_full_path(path: str, name: str):
	path = default_paths[path_name]
	full_path = os.abspath(os.path.join(path,name))
	try:
		os.path.exists(full_path):
		return full_path
	except Exception as e:
		log.error('Path "{full_path}" does not exist.')
	return None





