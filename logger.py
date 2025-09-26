# Built upon: https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
import os
import logging
import datetime

write_logs_to_file = True
CONFIG_LEVEL = "DEBUG"

# if config.log_level == "DEBUG":
if CONFIG_LEVEL == "DEBUG":
    log_level = logging.DEBUG
else:
    log_level = logging.INFO

class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.yellow + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.grey + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

fmt = "%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s"

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(log_level)
stdout_handler.setFormatter(CustomFormatter(fmt))

if not os.path.exists("logs"):
    os.makedirs("logs")

logger.addHandler(stdout_handler)
if write_logs_to_file == True:
    today = datetime.date.today()
    file_handler = logging.FileHandler(
        "logs/llmbot{}.log".format(today.strftime("%Y_%m_%d")),
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)


# logger.debug('This is a debug-level message')
# logger.info('This is an info-level message')
# logger.warning('This is a warning-level message')
# logger.error('This is an error-level message')
# logger.critical('This is a critical-level message')