# ---   IMPORTS   --- #
# ------------------- #
from typing import Union
import yaml
import logging
import logging.config
import sys
import os


# ---   LOGGER   --- #
# ------------------ #
# Global variable for the global logger
LOGGER: Union[logging.Logger, None] = None


def main_logger_init(rootdir=''):
    """
    Initialize the main logger (global variable LOGGER).

    :param rootdir: Path to the directory where the vl3d.py script is located.
    :type rootdir: str
    :return: Nothing.
    """
    global LOGGER
    with open(os.path.join(rootdir, "config/logging.yml"), "r") as file:
        yml = yaml.safe_load(file)
        logging.config.dictConfig(yml)
        LOGGER = logging.getLogger("logger_vl3d")
        LOGGER.debug("LOGGER was successfully initialized!")
