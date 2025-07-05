import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"

# Configure the root logger only once
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def get_logger(name=None):
    """
    Returns a logger instance with the specified name.

    Args:
        name (str, optional): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        None
    """
    return logging.getLogger(name)
