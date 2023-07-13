# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.inout.model_io import ModelIO
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING


# ---   CLASS   --- #
# ----------------- #
class ModelWriter(Writer):
    """
    :author: Alberto M. Esmoris Pena

    Class for writing tasks/operations (mostly to be used in pipelines) related
    to models.

    See :class:`.Writer`
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_writer_args(spec):
        """
        Extract the arguments to initialize/instantiate a ModelWriter
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a ModelWriter.
        """
        # Initialize from parent
        kwargs = Writer.extract_writer_args(spec)
        # Extract particular arguments of ModelWriter
        path = spec.get('out_model', None)
        if path is not None:  # Dont overload with not None path
            kwargs['path'] = path
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path=None):
        """
        Initialize/instantiate a ModelWriter.

        See :class:`.Writer` and :meth:`writer.Writer.__init__`
        """
        # Call parent's init
        super().__init__(path=path)

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, model, prefix=None, info=True):
        """
        Write the given model.

        :param model: The model to be written.
        :type model: :class:`.Model`
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path.
        :type prefix: str
        :param info: Whether to log an info message (True) or not (False).
        :type info: bool
        """
        # Prepare path and write
        path = self.prepare_path(prefix)
        ModelIO.write(model, path)
        # Log info if requested
        if info:
            LOGGING.LOGGER.info(f'Model written to "{path}"')
