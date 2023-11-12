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


def test_txt2img():
	global mm
	model = mm.get_loaded_model('stable_diffusion_v1_5.safetensors')
	assert(model is not None)
	prompt = pos_prompt + ' ### ' + neg_prompt
	images = mm.txt2img(model, prompt)
	assert(images is not None)
	mm.save_images(images)


