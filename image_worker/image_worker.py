#   image_worker.py


from queue import Queue
from image_worker.scripts.jobs import ImageJob
from model_manager.model_manager import ModelManager

class ImageWorker():
	def __init__(self, input_dir, output_dir, path_to_models):
		self.manager = ModelManager(input_dir, output_dir, path_to_models)
		self.jobs = Queue(maxsize = 20)
		self.completed_jobs = []
		self.failed_jobs = []

	def run(self):
		while(not self.jobs.empty()):
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
		if job.run(self.manager):
			return True
		return False


