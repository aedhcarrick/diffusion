# test_image_worker.py

from image_worker import ImageWorker

def test_simple_job():
	worker = ImageWorker()
	job = {
		"prompt": "a cow eating nachos",
		"sampler": "DDIM",
		"steps": 50,
		"height": 512,
		"width": 768,
		"guidance": 7.5,
		"seed": 42,
	}
	worker.submit_job('txt2img', job)
	worker.run()

