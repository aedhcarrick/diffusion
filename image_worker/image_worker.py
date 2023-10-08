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
		self.log.info(f'Stopping image worker....')
		self.running = False
		self.thread.join()
		self.log.info(f'Image worker stopped.')

	def run(self):
		while(self.running):
			self.log.info(f'Waiting for job....')
			time.sleep(5)
			cur_job = self.jobs.get()
			self.log.info(f'Job received: {cur_job.ID}.')
			success = self.run_job(cur_job)
			if success:
				self.completed_jobs.append(cur_job)
				self.log.info(f'Job successful: {cur_job.ID}.')
			else:
				self.failed_jobs.append(cur_job)
				self.log.error(f'Job failed: {cur_job.ID}.')

	def submit_job(self, oper_list: dict):
		job = ImageJob(oper_list)
		self.log.info(f'Adding job to queue: {job.ID}')
		self.jobs.put(job)

	def run_job(self, job: ImageJob):
		self.log.info(f'Running job:  {job.ID}')
		return job.run(self.manager, self.input_dir, self.output_dir)


