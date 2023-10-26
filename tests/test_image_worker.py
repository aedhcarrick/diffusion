# test_model_manager.py

from manager.model_manager import ModelManager


def test_simple_job():
	mm = ModelManager()
	mm.load_model('deliberate_v3.safetensors')
	assert(mm.model is not None)
	assert(mm.model.is_loaded())
	assert(mm.clip is not None)
	assert(mm.clip.is_loaded())
	assert(mm.vae is not None)
	assert(mm.vae.is_loaded())

