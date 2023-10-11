#   image_worker.py


import logging
import os
import time

from queue import Empty, Queue
from image_worker.scripts.jobs import ImageJob
from model_manager.model_manager import ModelManager
from threading import Thread
from utils import log_config


log = logging.getLogger(__name__)
log_config.setup_logging()


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
		self.current_job = None
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
			try:
				self.current_job = self.jobs.get(timeout=60)
			except Empty:
				continue
			self.log.info('Job received.')
			success = self.run_job(self.current_job)
			if success:
				self.completed_jobs.append(self.current_job)
				self.log.info(f'Job completed successfully: {self.current_job.job_ID}')
			else:
				self.failed_jobs.append(self.current_job)
				self.log.error(f'Job failed: {self.current_job.job_ID}')
			self.current_job = None

	def submit_job(self, oper_list: dict):
		job = ImageJob(oper_list)
		self.jobs.put(job)
		return job.job_ID

	def run_job(self, job: ImageJob):
		return job.run(self.manager, self.input_dir, self.output_dir)

	def check_job(self, job_ID):
		if self.current_job.job_ID == job_ID:
			return { 'sate': 'Working', 'progress': job.prog }
		for job in self.failed_jobs:
			if job.job_ID == job_ID:
				return { 'state': 'Failed', 'progess': job.prog }
		for job in self.completed_jobs:
			if job.job_ID == job_ID:
				return { 'state': 'Completed', 'progess': job.prog }


