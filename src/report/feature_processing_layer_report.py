# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class FeatureProcessingLayerReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports that represent a feature processing layer.

    See :class:`.Report`.
    See also :class:`.RBFFeatProcessingLayer`.

    :ivar M: The matrix of kernel's centers.
    :vartype M: :class:`np.ndarray`
    :ivar Omega: The matrix of kernel sizes (think about curvatures).
    :vartype Omega: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, M, Omega, **kwargs):
        """
        Initialize an instance of FeatureProcessingLayerReport.

        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.M = M
        self.Omega = Omega
        # Validate
        if self.M is None or len(self.M) < 1:
            raise ReportException(
                'FeatureProcessingLayerReport did not receive the centers of '
                'the kernel.'
            )
        if self.Omega is None or len(self.Omega) < 1:
            raise ReportException(
                'FeatureProcessingLayerReport did not receive the sizes of '
                'the kernel.'
            )

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (centers and sizes) to files.

        :param path: Path to the directory where the report files must be
            written.
        :type path: str
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Expand path if necessary
        if out_prefix is not None and path[0] == '*':
            path = out_prefix[:-1] + path[1:]
        # Check
        IOUtils.validate_path_to_directory(
            path,
            'Cannot find the directory to write the features processing '
            'layer representation'
        )
        # Output kernel's centers
        np.savetxt(
            os.path.join(path, 'M.csv'),
            self.M,
            fmt='%.7f'
        )
        # Output kernel's sizes
        np.savetxt(
            os.path.join(path, 'Omega.csv'),
            self.Omega,
            fmt='%.7f'
        )
