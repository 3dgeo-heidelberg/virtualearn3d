# ---   IMPORTS   --- #
# ------------------- #
from src.inout.point_cloud_io import PointCloudIO
from src.main.vl3d_exception import VL3DException


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
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path):
        """
        Initialize/instantiate a Writer

        :param path: The path itself. It might (or NOT) need a prefix. Any
            path that starts with "*" in the context of a Writer will need
            a prefix.
        """
        self.path = path

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, pcloud, prefix=None):
        """
        Write the given point cloud.

        :param pcloud: The point cloud to be written.
        :param prefix: If None, the writing applies to path. If not None,
            the writing applies to prefix+path
        """
        # Prepare path
        path = self.path
        if prefix is not None:
            path = prefix[:-1] + path[1:]
        # Write
        PointCloudIO.write(pcloud, path)

    # ---   CHECKS   --- #
    # ------------------ #
    def needs_prefix(self):
        """
        Check whether the Writer needs a prefix for write operations or not.

        :return: True if the Writer needs a prefix, False otherwise.
        """
        return self.path[0] == "*"

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_writer_class(spec):
        """
        Extract the miner's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a miner.
        :rtype: :class:`.Miner`
        """
        writer = spec.get('writer', None)
        if writer is None:
            raise ValueError(
                "Writing a point cloud requires a writer. None was specified."
            )
        # Check writer class
        writer_low = writer.lower()
        if writer_low == 'writer':
            return Writer
        # An unknown writer was specified
        raise ValueError(f'There is no known writer "{writer}"')