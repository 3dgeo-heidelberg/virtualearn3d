# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
import json


class JsonIO:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for input/output operations related to
    JSON files.
    """
    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read(path):
        """
        Read a JSON file.

        :param path: Path to the JSON file to be read.
        :return: Read JSON.
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find JSON file at given input path:'
        )
        # Read and return JSON
        with open(path) as jsonf:
            return json.load(jsonf)
