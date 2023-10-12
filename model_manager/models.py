#  models.py


class LoadedModel():
	def __init__(self, name, config, device, model):
		self.name = name
		self.base = 'sd15'
		self.config = config
		self.device = device
		self.model = model

