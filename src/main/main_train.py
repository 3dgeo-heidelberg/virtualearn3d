# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.model_io import ModelIO
from src.inout.io_utils import IOUtils
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
import os
import time


# ---   CLASS   --- #
# ----------------- #
class MainTrain:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for training tasks.
    """
    # ---  MAIN METHOD   --- #
    # ---------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for training tasks.

        :param spec: Key-word specification
        """
        LOGGING.LOGGER.info('Starting training ...')
        start = time.perf_counter()
        pcloud = PointCloudFactoryFacade.make_from_file(
            MainTrain.extract_input_path(spec)
        )
        model_class = MainTrain.extract_model_class(spec)
        model = model_class(**model_class.extract_model_args(spec))
        model = model.train(pcloud)
        ModelIO.write(model, MainTrain.extract_output_path(spec))
        end = time.perf_counter()
        LOGGING.LOGGER.info(f'Training computed in {end-start:.3f} seconds.')

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_input_path(spec):
        """
        Extract the input path from the key-word specification.

        :param spec: The key-word specification.
        :return: Input path as string.
        :rtype: str
        """
        path = spec.get('in_pcloud', None)
        if path is None:
            raise ValueError(
                "Training a model requires an input point cloud. "
                "None was given."
            )
        IOUtils.validate_path_to_file(
            path,
            'Cannot find the input file for model training.\n'
            f'Given path: {path}'
        )
        return path

    @staticmethod
    def extract_output_path(spec):
        """
        Extract the output path from the key-word specification

        :param spec: The key-word specification.
        :return: Output path as string.
        :rtype: str
        """
        path = spec.get('out_model', None)
        if path is None:
            raise ValueError(
                'Training a model requires an output path to store the '
                'trained model. None was given.'
            )
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the trained model:'
        )
        return path

    @staticmethod
    def extract_model_class(spec):
        """
        Extract the model's class from the key-word specification.

        :param spec: The key-word specification.
        :return:
        """
        model = spec.get('train', None)
        if model is None:
            raise ValueError(
                "Training a model requires a model. None was specified."
            )
        # Check model class
        model_low = model.lower()
        if model_low == 'randomforestclassifier':
            return RandomForestClassificationModel
        # An unknown model was specified
        raise ValueError(f'There is no known model "{model}"')
