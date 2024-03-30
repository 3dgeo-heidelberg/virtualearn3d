# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
from src.pipeline.predictive_pipeline import PredictivePipeline
from src.model.deeplearn.arch import architecture
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
    def read_predictive_pipeline(path, new_nn_path=None):
        """
        Read a predictive pipeline file.

        :param path: Path to the pipeline file.
        :param new_nn_path: Path to the serialized neural network when loading
            predictive pipelines with deep learning models. It can be None, in
            which case the `nn_path` attribute of the serialized
            :class:`.Architecture` will be considered.
        :type path: str
        :return:
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find model file at given input path:'
        )
        # Read and return predictive pipeline
        architecture.new_nn_path = new_nn_path
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
        # Write external model file for deep learning models
        if predictive_pipeline.is_using_deep_learning():
            predictive_pipeline.write_deep_learning_model(
                f'{path[:path.rfind(".")]}.model'
            )
        # Write output pipeline
        joblib.dump(predictive_pipeline, path, compress=True)
