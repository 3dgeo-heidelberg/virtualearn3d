# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.inout.model_io import ModelIO


# ---   CLASS   --- #
# ----------------- #
class ModelWriter(Writer):
    """
    :author: Alberto M. Esmoris Pena

    Class for writing tasks/operations (mostly to be used in pipelines) related
    to models.

    See :class:`.Writer`
    """
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

