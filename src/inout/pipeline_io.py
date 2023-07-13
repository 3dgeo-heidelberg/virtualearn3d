# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
from src.pipeline.predictive_pipeline import PredictivePipeline
import joblib
import os


# ---   CLASS   --- #
# ----------------- #
class PipelineIO:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for input/output operations related to
    pipelines.
    """
    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read_predictive_pipeline(path):
        """
        Read a predictive pipeline file.

        :param path: Path to the pipeline file.
        :type path: str
        :return:
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find model file at given input path:'
        )
        # Read and return predictive pipeline
        predictive_pipeline = joblib.load(path)
        if not isinstance(predictive_pipeline, PredictivePipeline):
            raise TypeError(
                'Reaf file does not represent a predictive pipeline:\n'
                f'"{path}"'
            )
        return predictive_pipeline

    # ---  WRITE / STORE  --- #
    # ----------------------- #
    @staticmethod
    def write_predictive_pipeline(predictive_pipeline, path):
        """
        Write a predictive pipeline to a file.

        :param predictive_pipeline: The predictive pipeline to be written.
        :type predictive_pipeline: :class:`.PredictivePipeline`
        :param path: Path where the pipeline file must be written.
        :type path: str
        :return: Nothing.
        """
        # Validate output directory
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'The parent of the output pipeline path is not a directory:'
        )
        # Validate predictive pipeline
        if not isinstance(predictive_pipeline, PredictivePipeline):
            raise TypeError(
                'Given object will not be written because it is not a '
                'predictive pipeline.'
            )
        # Write output pipeline
        joblib.dump(predictive_pipeline, path, compress=True)
