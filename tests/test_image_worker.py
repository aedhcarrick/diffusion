# test_image_worker.py

from image_worker.image_worker import ImageWorker


def test_simple_job():
	input_dir = 'inputs'
	output_dir = 'outputs'
	model_dir = 'models'
	worker = ImageWorker(input_dir, output_dir, model_dir)
	job = {
		"job_type": "image",
		"operations": [
			{
			"oper_type": "txt2img",
			"prompt": "a cow eating nachos",
			"sampler": "DDIM",
			"steps": 50,
			"height": 512,
			"width": 768,
			"guidance": 7.5,
			"seed": 42,
			"batch_size": 1,
			}
		]
	}
	worker.submit_job(job)
	worker.run()

