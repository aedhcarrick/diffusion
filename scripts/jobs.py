# workflow.py


from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from typing import Literal, Union
import uuid


class Sampler():
	def __init__(self, name: Literal["DPM", "PLMS", "DDIM"]):
		self.name = name

	def get_sampler(self, model):
		if self.name == "DPM":
			return DPMSolverSampler(model)
		elif self.name == "PLMS":
			return PLMSSampler(model)
		else:
			return DDIMSampler(model)


class Options():
	def __init__(self, settings: dict):
		self.prompt: str = "a painting of a dog eating nachos"
		self.output_dir: str = "outputs"
		self.config: str = "configs/stable-diffusion/v1-inference.yaml"
		self.model: str = "models/checkpoints/model.ckpt"
		self.sampler: Union[None, Sampler] = None
		self.iter: int = 2
		self.steps: int = 50
		self.eta: float = 0.0
		self.height: int = 512
		self.width: int = 512
		self.channels: int = 4
		self.down_sampling: int = 8
		self.guidance: float = 7.5
		self.batch_size: int = 4
		self.seed: int = 42
		self.precision: Literal["full", "autocast"] = "autocast"
		for key, value in settings.items():
			if key == 'sampler':
				self.set_sampler(settings['sampler'])
			if hasattr(self, key):
				setattr(self, key, value)

	def set_sampler(self, samp_name: str):
		self.sampler = Sampler(samp_name)


class ImageJob():
	def __init__(self, job_type: Literal["txt2img"]):
		self.job_type = job_type
		self.job_ID: str = str(uuid.uuid4())
		self.options: Union[None, Options] = None







