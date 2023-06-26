# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.model_io import ModelIO
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
        Entry point logic for training tasks

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
