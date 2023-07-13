# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.utils.dict_utils import DictUtils
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PredictionsWriter(Writer):
    """
    :author: Alberto M. Esmoris Pena

    Class for writing predictions (mostly to be used in pipelines).

    See :class:`.Writer`.
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_writer_args(spec):
        """
        Extract the arguments to initialize/instantiate a PredictionsWriter
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a PredictionsWriter.
        """
        # Initialize from parent
        kwargs = Writer.extract_writer_args(spec)
        # Extract particular arguments of PredictionsWriter
        path = spec.get('out_preds', None)
        if path is not None:  # Dont overload with None path
            kwargs['path'] = path
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path=None):
        """
        Initialize/instantiate a PredictionsWriter.

        See :class:`.Writer` and :meth:`writer.Writer.__init__`.
        """
        # Call parent's init
        super().__init__(path=path)

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, preds, prefix=None, info=True):
        """
        Write the given predictions.

        :param preds: The predictions to be written.
        :type preds: :class:`np.ndarray`
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path.
        :type prefix: str
        :param info: Whether to log an info message (True) or not (False).
        :type info: bool
        """
        # Prepare path and write
        path = self.prepare_path(prefix)
        fmt = '%d' if np.issubdtype(preds.dtype, np.integer) else '%.8f'
        np.savetxt(path, preds, fmt=fmt)
