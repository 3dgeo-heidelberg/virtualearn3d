# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
import os


# ---   CLASS   --- #
# ----------------- #
class PwiseActivationsReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to point-wise activations.
    See :class:`.Report`.

    :ivar X: The matrix of coordinates representing the point cloud.
    :vartype X: :class:`np.ndarray`
    :ivar activations: The matrix of features representing the point-wise
        activations.
    :vartype activations: :class:`np.ndarray`
    :ivar y: The vector of expected classes.
    :vartype y: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of PwiseActivationsReport

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *X* (``np.ndarray``) --
                The matrix of coordinates representing the point cloud.
            *   *activations* (``np.ndarray``) --
                The matrix of features representing the point-wise activations.
            *   *y* (``np.ndarray``) --
                The expected classes for each point in :math:`\pmb{X}`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.X = kwargs.get('X', None)
        self.activations = kwargs.get('activations', None)
        self.y = kwargs.get('y', None)

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (point cloud) to a file (LAZ).
        :param path: Path to the file where the report must be written.
        :type path: str
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Expand path if necessary
        if out_prefix is not None and path[0] == "*":
            path = out_prefix[:-1] + path[1:]
        # Check
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the point-wise activations:'
        )
        # Write
        PointCloudIO.write(
            PointCloudFactoryFacade.make_from_arrays(
                self.X,
                self.activations,
                self.y
            ),
            path
        )
        # Log
        LOGGING.LOGGER.info(f'Point-wise activations written to "{path}"')
