# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.sequential_pipeline import SequentialPipeline
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class MainPipeline:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for pipelines
    """
    # ---  MAIN METHOD   --- #
    # ---------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for pipelines.

        :param spec: Key-word specification.
        """
        LOGGING.LOGGER.info('Starting pipeline ...')
        start = time.perf_counter()
        pipeline_class = MainPipeline.extract_pipeline_class(spec)
        pipeline = pipeline_class(**spec)
        pipeline.run()
        end = time.perf_counter()
        LOGGING.LOGGER.info(f'Pipeline computed in {end-start:.3f} seconds')

    # ---  EXTRACT FORM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_pipeline_class(spec):
        """
        Extract the pipeline's class from the Key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a pipeline.
        :rtype: :class:`.Pipeline`
        """
        # Check pipeline class
        pipeline = spec.get("sequential_pipeline", None)
        if pipeline is not None:
            return SequentialPipeline
        # No pipeline detected
        raise ValueError('No pipeline was detected in the given specification')
