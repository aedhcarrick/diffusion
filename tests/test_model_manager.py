# tests/test_model_manager.py

from manager.model_manager import ModelManager

global mm
mm = ModelManager()

pos_prompt = "A painting of a dog eating nachos, in the style of Caravaggio"
neg_prompt = "text, watermark, jpeg artifacts"
clip_skip = 1
steps = 50
sampler = 'DDIM'
scheduler = 'normal'
cfg = 7.5
denoise = 1.0


def test_load_model():
	global mm
	mm.load_model('stable_diffusion_v1_5.safetensors')
	assert(mm.model is not None)
	assert(mm.model.is_loaded())
	assert(mm.clip is not None)
	assert(mm.clip.is_loaded())
	assert(mm.vae is not None)
	assert(mm.vae.is_loaded())

def test_sampler():
	global mm
	cond = mm.get_cond(pos_prompt)
	mm.sample(cond=cond, batch_size=1)
	assert(mm.sample is not None)


