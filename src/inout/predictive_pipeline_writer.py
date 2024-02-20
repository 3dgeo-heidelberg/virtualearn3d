# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.inout.pipeline_io import PipelineIO
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING


# ---   CLASS   --- #
# ----------------- #
class PredictivePipelineWriter(Writer):
    """
    :author: Alberto M. Esmoris Pena

    Class for writing predictive pipelines.

    See :class:`.PredictivePipeline`.
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_writer_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        PredictivePipelineWriter from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            PredictivePipelineWriter.
        """
        # Initialize from parent
        kwargs = Writer.extract_writer_args(spec)
        # Extract particular arguments of PredictivePipelineWriter
        path = spec.get('out_pipeline', None)
        if path is not None:  # Dont overload with not None path
            kwargs['path'] = path
        kwargs['include_writer'] = spec.get('include_writer', None)
        kwargs['include_imputer'] = spec.get('include_imputer', None)
        kwargs['include_feature_transformer'] = spec.get(
            'include_feature_transformer', None
        )
        kwargs['include_class_transformer'] = spec.get(
            'include_class_transformer', None
        )
        kwargs['include_miner'] = spec.get('include_miner', None)
        kwargs['ignore_predictions'] = spec.get('ignore_predictions', None)
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path=None, **kwargs):
        """
        Initialize/instantiate a PredictivePipelineWriter.

        See :class:`.Writer` and :meth:`writer.Writer.__init__`
        """
        # Call parent's init
        super().__init__(path=path)
        # Assign attributes of PredictivePipelineWriter
        self.include_writer = kwargs.get('include_writer', False)
        self.include_imputer = kwargs.get('include_imputer', True)
        self.include_ftransf = kwargs.get('include_feature_transformer', True)
        self.include_ctransf = kwargs.get('include_class_transformer', True)
        self.include_miner = kwargs.get('include_miner', True)
        self.ignore_predictions = kwargs.get('ignore_predictions', False)

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, pipeline, prefix=None, info=True):
        """
        Write the predictive version of the given pipeline.

        :param pipeline: Pipeline (must be transformable to a predictive
            pipeline).
        :type pipeline: :class:`.Pipeline`
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path.
        :type prefix: str
        :param info: Whether to log an info message (True) or not (False).
        :type info: bool
        """
        # Prepare path and write
        path = self.prepare_path(prefix)
        PipelineIO.write_predictive_pipeline(
            pipeline.to_predictive_pipeline(
                include_writer=self.include_writer,
                include_imputer=self.include_imputer,
                include_feature_transformer=self.include_ftransf,
                include_class_transformer=self.include_ctransf,
                include_miner=self.include_miner,
                ignore_predictions=self.ignore_predictions
            ),
            path
        )
        # Log info if requested
        if info:
            LOGGING.LOGGER.info(f'Predictive pipeline written to "{path}"')
