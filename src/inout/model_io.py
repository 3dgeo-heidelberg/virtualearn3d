# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
from src.model.model import Model
import joblib
import os


class ModelIO:
    """
    :author: Alberto M. Esmoris Pena

    Class with utils static methods for input/output operations related to
    models.
    """
    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read(path):
        """
        Read a model file.

        :param path: Path to the model file.
        :return: Instance of Model (or corresponding derived class)
        :rtype: :class:`.Model`
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find model file at given input path:'
        )
        # Read and return model
        model = joblib.load(path)
        if not isinstance(model, Model):
            raise TypeError(
                'Read file does not represent a model:\n'
                f'"{path}"'
            )
        return model

    # ---  WRITE / STORE  --- #
    # ----------------------- #
    @staticmethod
    def write(model, path):
        """
        Write a model to a file.

        :param model: The model to be written.
        :param path: Path where the model file must be written.
        :return: Nothing
        """
        # Validate output directory
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'The parent of the output model path is not a directory:'
        )
        # Validate model
        if not isinstance(model, Model):
            raise TypeError(
                'Given object will not be written because it is not a model.'
            )
        # Write output model
        joblib.dump(model, path)
