# ---   IMPORTS   --- #
# ------------------- #
from typing import Union
import yaml
import logging
import logging.config


# ---   LOGGER   --- #
# ------------------ #
# Global variable for the global logger
LOGGER: Union[logging.Logger, None] = None


def main_logger_init():
    """
    Initialize the main logger (global variable LOGGER)

    :return: Nothing
    """
    global LOGGER
    with open("config/logging.yml", "r") as file:
        yml = yaml.safe_load(file)
        logging.config.dictConfig(yml)
        LOGGER = logging.getLogger("logger_vl3d")
        LOGGER.debug("LOGGER was successfully initialized!")
