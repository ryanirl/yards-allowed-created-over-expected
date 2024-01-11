import logging
import sys


def setup_logger():
    DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    basic_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a console handle for command line output.
    console_handler = logging.StreamHandler(stream = sys.stdout)
    console_handler.setFormatter(basic_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


