# test_logger.py


import logging
from utils import log_config

log = logging.getLogger(__name__)
log_config.setup_logging()


class FakeClass():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(log_config.ThreadContextFilter())
		self.log.info(f'Instance of {self.__class__.__name__} created.')

def test_logger():
	log.debug("Debug Message")
	log.info("Info Message")
	log.warning("Info Warning")
	log.error("Error Message")
	log.critical("Critical Message")

	fake = FakeClass()


