#  utils/paths.py


default_paths = {
	'checkpoints': 'models/checkpoints',
	'input_dir': 'input',
	'output_dir': 'output',
}


import os


def get_path(path_name: str):
	return default_paths[path_name]

def get_full_path(path: str, name: str):
	path = default_paths[path]
	full_path = os.path.abspath(os.path.join(path,name))
	if os.path.exists(full_path):
		return full_path
	else:
		return None





