#   image_worker.py


from queue import Queue
from scripts.jobs import ImageJob, Options
from scripts.txt2img import txt2img


class ImageWorker():
	def __init__(self):
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

	def submit_job(self, job_type: str, job: dict):
		options = Options(job)
		job = ImageJob(job_type)
		job.options = options
		self.jobs.put(job)

	def run_job(self, job: ImageJob):
		if job.job_type == 'txt2img':
			return txt2img(job.options)
