# tests/test_model_manager.py

from manager.model_manager import ModelManager

global mm
global model
global clip
global vae

mm = ModelManager()

def test_model_load():
	global mm
	global model
	global clip
	global vae
	model, clip, vae = mm.load_model('deliberate_v3.safetensors')
	assert(model is not None)
	assert(model.is_loaded())
	assert(clip is not None)
	assert(clip.is_loaded())
	assert(vae is not None)
	assert(vae.is_loaded())

