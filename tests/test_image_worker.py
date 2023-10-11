# test_image_worker.py

from image_worker.image_worker import ImageWorker


def test_simple_job():
	input_dir = 'input'
	output_dir = 'output'
	model_dir = 'models'
	worker = ImageWorker(input_dir, output_dir, model_dir)

	start_count = len(os.listdir(output_dir))
	worker.start()

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
	worker.stop()
	final_count = len(os.listdir(output_dir))
	assert(final_count > start_count)

