# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.inout.io_utils import IOUtils
import src.main.main_logger as LOGGING
import os


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ReportException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to report components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Report:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface governing any report.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Root initialization for any instance of type Report.

        :param kwargs: The attributes for the report.
        """
        pass

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the report.

        :return: String representation of the report.
        :rtype: str
        """
        raise ReportException(
            f'{__class__} extending Report does not provide a valid '
            'implementation for the __str__ method.'
        )

    def to_string(self):
        """
        Wrapper for :meth:`report.Report.__str__`.
        """
        return self.__str__()

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path):
        """
        Write the report to a file.

        :param str path: Path to the file where the report must be written.
        :return: Nothing, the output is written to a file.
        """
        # Check
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the report:'
        )
        # Write
        with open(path, 'w') as outf:
            outf.write(self.to_string())
        # Log
        LOGGING.LOGGER.info(f'Report written to "{path}"')
