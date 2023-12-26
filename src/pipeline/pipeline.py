# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import src.main.main_logger as LOGGING
from src.main.vl3d_exception import VL3DException
from src.inout.io_utils import IOUtils
import os


# ---  EXCEPTIONS  --- #
# -------------------- #
class PipelineException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to pipeline components.
    See :class:`VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Pipeline:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface for any pipeline and a common
    baseline implementation.

    :ivar in_pcloud: Either a string or a list of strings representing paths to
        input point clouds.
    :vartype in_pcloud: str or list
    :ivar in_pcloud_concat: An alternative to in_pcloud that concatenates
        many point clouds and supports conditional filters. It must be a list
        of dictionaries each specifying an input point cloud and potentially
        a list of conditions.
    :vartype in_pcloud_concat: list
    :ivar out_pcloud: Either a string or a list of strings representing paths
        to output point clouds. Any output string that ends with and asterisk
        "*" will be used as a prefix for outputs specified in the components
        of the pipeline.
    :vartype out_pcloud: str or list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handles the root-level (most basic) initialization of any pipeline.

        :param kwargs: The attributes for the Pipeline.
        """
        # Input point clouds
        self.in_pcloud = kwargs.get("in_pcloud", None)
        if self.in_pcloud is not None:  # Validate the paths to input point clouds
            if isinstance(self.in_pcloud, (list, tuple)):  # Many paths
                for path in self.in_pcloud:
                    if path is None:
                        raise PipelineException(
                            'Pipelines do not support lists of input point '
                            'cloud paths with null elements.'
                        )
                    IOUtils.validate_path_to_file(
                        path,
                        "Pipeline received an invalid input point cloud path "
                        "in the list:",
                        accept_url=True
                    )
            else:  # Path as string
                IOUtils.validate_path_to_file(
                    self.in_pcloud,
                    "Pipeline received an invalid input point cloud path:"
                )
        # Alternative : Input point clouds concatenation
        self.in_pcloud_concat = kwargs.get('in_pcloud_concat', None)
        if self.in_pcloud is not None and self.in_pcloud_concat is not None:
            raise PipelineException(
                'Pipeline received both in_pcloud and in_pcloud_concat. '
                'This is ambiguous and thus not supported.\n'
                'Please, choose either in_pcloud or in_pcloud_concat but '
                'avoid using both at the same time.'
            )
        # TODO Rethink : Anything else to do here?
        # Handle what happens when no input has been specified
        if self.in_pcloud is None and self.in_pcloud_concat is None:
            raise PipelineException(
                'It is not possible to build a pipeline without at least one '
                'input point cloud.'
            )
        # Output point clouds
        self.out_pcloud = kwargs.get('out_pcloud', None)
        if self.out_pcloud is None:  # Info about no output point clouds
            LOGGING.LOGGER.info(
                'A pipeline has been built with no output point clouds.'
            )
        else:  # Validate the paths to output point clouds
            if isinstance(self.out_pcloud, (list, tuple)):  # Many paths
                for path in self.out_pcloud:
                    if path is None:
                        raise PipelineException(
                            'Pipelines do not support lists of output point '
                            'cloud paths with null elements.'
                        )
                    IOUtils.validate_path_to_directory(
                        os.path.dirname(path),
                        "Pipeline received an output point cloud path with an "
                        "invalid parent directory in the list: "
                    )
            else:  # Path as string
                IOUtils.validate_path_to_directory(
                    self.out_pcloud,
                    "Pipeline received an output point cloud path with an "
                    "invalid parent directory:"
                )
        # Validate input and output dimensionalities match
        num_inputs, num_outputs = 1, 1
        if isinstance(self.in_pcloud, (list, tuple)):
            num_inputs = len(self.in_pcloud)
        if isinstance(self.out_pcloud, (list, tuple)):
            num_outputs = len(self.out_pcloud)
        if num_inputs != num_outputs:
            raise PipelineException(
                f'Pipeline with input dimensionality ({num_inputs}) distinct '
                f'from output dimensionality ({num_outputs}) is not '
                'supported.'
            )



    # ---  RUN PIPELINE  --- #
    # ---------------------- #
    @abstractmethod
    def run(self):
        """
        Run the pipeline.

        :return: Nothing.
        """
        pass

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def to_predictive_pipeline(self, **kwargs):
        """
        Transforms the current pipeline to a predictive pipeline, if possible.
        See :class:`.PredictivePipeline`.

        :return: A predictive pipeline wrapping this pipeline and providing a
            predictive strategy.
        :rtype: :class:`.PredictivePipeline`.
        """
        raise PipelineException(
            f'{self.__class__.__name__} cannot be transformed to a predictive '
            'pipeline.'
        )

    def is_using_deep_learning(self):
        """
        Check whether the pipeline uses deep learning or not.

        By default, pipelines do not support deep learning. Any pipeline
        that supports deep learning models must explicitly overload this method
        to return True.

        :return: True if the pipeline uses deep learning, false otherwise.
        :rtype: bool
        """
        return False

    def write_deep_learning_model(self, path):
        """
        Write the deep learning model used in the pipeline to disk.

        :param path: Path where the deep learning model must be written.
        :type path: str
        """
        raise PipelineException(
            f'{self.__class__.__name__} does not support exporting deep '
            'learning models.'
        )
