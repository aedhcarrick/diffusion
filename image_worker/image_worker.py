#   image_worker.py


import os
import time

from queue import Queue
from image_worker.scripts.jobs import ImageJob
from model_manager.model_manager import ModelManager
from threading import Thread


class ImageWorker():
	def __init__(self, input_dir, output_dir, path_to_models):
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
		self.running = True
		self.thread.start()

	def stop(self):
		self.running = False
		self.thread.join()

	def run(self):
		while(self.running):
			time.sleep(5)
			cur_job = self.jobs.get()
			success = self.run_job(cur_job)
			if success:
				self.completed_jobs.append(cur_job)
			else:
				self.failed_jobs.append(cur_job)

	def submit_job(self, oper_list: dict):
		job = ImageJob(oper_list)
		self.jobs.put(job)

	def run_job(self, job: ImageJob):
		return job.run(self.manager, self.input_dir, self.output_dir):


