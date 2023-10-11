#   image_worker.py


import logging
import os
import time

from queue import Queue
from image_worker.scripts.jobs import ImageJob
from model_manager.model_manager import ModelManager
from threading import Thread
from utils import log_config


class ImageWorker():
	def __init__(self, input_dir, output_dir, path_to_models):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(log_config.ThreadContextFilter())
		self.log.info('Initializing image worker..')
		self.manager = ModelManager(path_to_models)
		self.input_dir = input_dir
		self.output_dir = output_dir
		os.makedirs(self.input_dir, exist_ok=True)
		os.makedirs(self.output_dir, exist_ok=True)

		self.jobs = Queue(maxsize = 20)
		self.completed_jobs = []
		self.failed_jobs = []

		self.thread = Thread(target=self.run, daemon=True, args=())
		self.running = False

	def start(self):
		self.log.info(f'Starting image worker.')
		self.running = True
		self.thread.start()

	def stop(self):
		self.log.info(f'Stopping image worker...')
		self.running = False
		self.thread.join()
		self.log.info(f'Stopped.')

	def run(self):
		while(self.running):
			self.log.info(f'Waiting for jobs', end='')
			if self.jobs.empty():
				time.sleep(10)
				self.log.info(f'.', end='')
				continue
			cur_job = self.jobs.get()
			self.log.info('')
			self.log_info('Job received.')
			success = self.run_job(cur_job)
			if success:
				self.completed_jobs.append(cur_job)
				self.log.info(f'Job completed successfully: {cur_job.job_ID}')
			else:
				self.failed_jobs.append(cur_job)
				self.log.error(f'Job failed: {cur_job.job_ID}')

	def submit_job(self, oper_list: dict):
		job = ImageJob(oper_list)
		self.jobs.put(job)

	def run_job(self, job: ImageJob):
		return job.run(self.manager, self.input_dir, self.output_dir)


