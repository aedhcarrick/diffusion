# test_image_worker.py

from image_worker.image_worker import ImageWorker

operation = {
    "network_module": None,
    "network_weight": None,
    "network_mul": 1.0,
    "network_args": None,
    "ckpt": None,
    "vae": None,
    "xformers": True,
    "W": 512,
    "H": 512,
    "seed": None,
    "scale": 7.5,
    "sampler": 'DDIM',
    "steps": 40,
    "batch_size": 1,
    "clip_skip": None,
    "prompt": 'a dog eating nachos, in the style of Caravaggio',
}

global job = {
	"job_type": "image",
	"operations": [operation]
}


def test_start_worker():
	input_dir = 'inputs'
	output_dir = 'outputs'
	model_dir = 'models'
	worker = ImageWorker(input_dir, output_dir, model_dir)
	worker.start()


def test_simple_job():
	global job
	worker.submit_job(job)


def test_stop_worker():
	worker.stop()




'''
job template


operation
