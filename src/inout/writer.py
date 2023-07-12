# ---   IMPORTS   --- #
# ------------------- #
from src.inout.point_cloud_io import PointCloudIO
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils


# ---   EXCEPTIONS   --- #
# ---------------------- #
class WriterException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to writing components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Writer:
    """
    :author: Alberto M. Esmoris Pena

    Class for writing tasks/operations (mostly to be used in pipelines)

    :ivar path: The path to the output file for writing operations.
    :vartype path: str
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_writer_args(spec):
        """
        Extract the arguments to initialize/instantiate a Writer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a Writer.
        """
        # Extract particular arguments of Writer
        # Initialize
        kwargs = {
            'path': spec.get('out_pcloud', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path=None):
        """
        Initialize/instantiate a Writer

        :param path: The path itself. It might (or NOT) need a prefix. Any
            path that starts with "*" in the context of a Writer will need
            a prefix.
        """
        if path is None:
            raise WriterException(
                'Instantiating a Writer without a path is not supported.'
            )
        self.path = path

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, pcloud, prefix=None):
        """
        Write the given point cloud.

        :param pcloud: The point cloud to be written.
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path.
            See :meth:`writer.Writer.prepare_path`.
        """
        # Prepare path and write
        path = self.prepare_path(prefix)
        PointCloudIO.write(pcloud, path)

    def prepare_path(self, prefix):
        """
        Merge the path with the prefix to obtain the actual path for the
        writing.

        :return: Prepared path.
        :rtype: str
        """
        path = self.path
        if prefix is not None:
            path = prefix[:-1] + path[1:]
        return path

    # ---   CHECKS   --- #
    # ------------------ #
    def needs_prefix(self):
        """
        Check whether the Writer needs a prefix for write operations or not.

        :return: True if the Writer needs a prefix, False otherwise.
        """
        return self.path[0] == "*"
