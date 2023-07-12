# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.inout.model_io import ModelIO
from src.utils.dict_utils import DictUtils


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
        path = spec.get('out_path', None)
        if path is not None:  # Dont overload a not None path from ancestors
            kwargs['path'] = path
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path):
        """
        Initialize/instantiate a ModelWriter

        See :class:`.Writer` and :meth:`writer.Writer.__init__`
        """
        # Call parent's init
        super().__init__(path)

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, model, prefix=None):
        """
        Write the given model.

        :param model: The model to be written.
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path.
        """
        # Prepare path and write
        path = self.prepare_path(prefix)
        ModelIO.write(model, path)
