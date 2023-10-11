#  logging.py

import os
import sys
import json
import logging
import logging.config
import getpass
import threading

# custom formatter isn't working?

class ColoredFormatter(logging.Formatter):
	grey = "\x1b[0;37m"
	green = "\x1b[1;32m"
	yellow = "\x1b[33;21m"
	red = "\x1b[31;21m"
	bold_red = "\x1b[31;1m"
	reset = "\x1b[0m"

	def __init__(self, fmt, datefmt):
		super().__init__()
		self.format = fmt
		self.datefmt = datefmt
		self.FORMATS = {
			logging.DEBUG: grey + self.format + reset,
			logging.INFO: green + self.format + reset,
			logging.WARNING: yellow + self.format + reset,
			logging.ERROR: red + self.format + reset,
			logging.CRITICAL: bold_red + self.format + reset
		}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)


default_config = {
	"version": 1,
	"disable_existing_loggers": False,

	"formatters": {
		"basic": {
			(): ColoredFormatter,
			"format": "%(asctime)s - %(name)s - %(levelname)8s : %(message)s",
			"datefmt": "%m/%d/%Y %I:%M:%S %p"
		},
		"log_format": {
			"format": "%(asctime)s - %(name)s - %(levelname)8s : %(message)s",
			"datefmt": "%m/%d/%Y %I:%M:%S %p"
        },
		"json": {
			"format": "name: %(name)s, level: %(levelname)s, time: %(asctime)s, message: %(message)s"
		},
	},

	"handlers": {
		"console": {
			"class": "logging.StreamHandler",
			"level": "INFO",
			"formatter": "basic",
			"stream": "ext://sys.stdout"
		},
		"local_file_handler": {
			"class": "logging.handlers.RotatingFileHandler",
			"level": "DEBUG",
			"formatter": "log_format",
			"filename": "debug.log",
			"maxBytes": 1048576,
			"backupCount": 20,
			"encoding": "utf8",
			"delay" : True
		},
	},

	"loggers": {
		"classes": {
			"level": "INFO",
			"propagate": True
		},
		"message": {
			"level": "INFO",
			"propagate": True,
			"handlers": ["console"]
		}
	},

	"root": {
		"level": "INFO",
		"handlers": ["console","local_file_handler"],
	}
}


class ThreadContextFilter(logging.Filter):
	"""A logging context filter to add thread name and ID."""
	def filter(self, record):
		record.thread_id = str(threading.current_thread().ident)
		record.thread_name = str(threading.current_thread().name)
		return True


def setup_logging(
		default_log_config=None,
		default_level=logging.INFO,
		env_key='LOG_CFG'
		):
	dict_config = None
	logconfig_filename = default_log_config
	env_var_value = os.getenv(env_key, None)

	if env_var_value is not None:
		logconfig_filename = env_var_value

	if default_config is not None:
		dict_config = default_config

	if logconfig_filename is not None and os.path.exists(logconfig_filename):
		with open(logconfig_filename, 'rt') as f:
			file_config = json.load(f)
		if file_config is not None:
			dict_config = file_config

	if dict_config is not None:
		logging.config.dictConfig(dict_config)
	else:
		logging.basicConfig(level=default_level)


#	For Testing  #######################################################
class FakeClass():
	def __init__(self):
		self.log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
		self.log.addFilter(ThreadContextFilter())
		self.log.info(f'Instance of {self.__class__.__name__} created.')

def test_logger():
	log = logging.getLogger(__name__)
	setup_logging()
	log.debug("Debug Message")
	log.info("Info Message")
	log.warning("Info Warning")
	log.error("Error Message")
	log.critical("Critical Message")

	fake = FakeClass()

if __name__ == '__main__':
	test_logger()

