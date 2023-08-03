# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class ClassifiedPcloudReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to classified point clouds.
    See :class:`.Report`

    # TODO Rehtink: Doc ivars
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of ClassifiedPcloudReport.

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *X* (``np.ndarray``) --
                The matrix of coordinates representing the point cloud.
            *   *y* (``np.ndarray``) --
                The expected classes for each point in :math:`\pmb{X}`.
            *   *yhat* (``np.ndarray``) --
                The vector of point-wise predictions.
            *   *zhat* (``np.ndarray``) --
                The matrix of point-wise softmax (row-points, col-classes).
            *   *class_names* (``np.ndarray``) --
                The name for each class. If not given, they will be considered
                as C1, ..., CN by default.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)
        self.yhat = kwargs.get('yhat', None)
        self.zhat = kwargs.get('zhat', None)
        self.class_names = kwargs.get('class_names', None)
        if self.class_names is None:
            self.class_names = [f'C{i}' for i in range(self.zhat.shape[1])]

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
            'Cannot find the directory to write the classified point cloud:'
        )
        # Write
        fnames = ['Prediction']
        feats = self.yhat.reshape(-1, 1)
        if self.zhat is not None:
            fnames = fnames + self.class_names
            feats = np.hstack([feats, self.zhat])
        PointCloudIO.write(
            PointCloudFactoryFacade.make_from_arrays(
                self.X,
                feats,
                self.y,
                fnames=fnames
            ),
            path
        )
        # Log
        LOGGING.LOGGER.info(f'Classified point cloud written to "{path}"')
