# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.main.main_mine import MainMine
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.model_io import ModelIO
from src.inout.io_utils import IOUtils
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
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
        See :class:`.MainMine` and
        :func:`main_mine.MainMine.extract_input_path`.
        """
        return MainMine.extract_input_path(
            spec,
            none_path_msg='Training a model requires an input point cloud. '
                          'None was given.',
            invalid_path_msg='Cannot find the input file for model training.\n'
                             'Given path: {path}'
        )

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
        if model_low == 'pointnetpwiseclassifier':
            return PointNetPwiseClassifModel
        # An unknown model was specified
        raise ValueError(f'There is no known model "{model}"')
