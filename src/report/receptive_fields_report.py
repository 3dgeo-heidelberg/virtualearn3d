# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldsReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to receptive fields.
    See :class:`.Report`.

    :ivar X_rf: The matrix of coordinates for each receptive field.
    :vartype X_rf: :class:`np.ndarray`
    :ivar zhat_rf: The softmax scores for the predictions on each receptive
        field.
    :vartype zhat_rf: :class:`np.ndarray`
    :ivar yhat_rf: The predictions for each receptive field.
    :vartype yhat_rf: :class:`np.ndarray`
    :ivar y_rf: The expected values for each receptive field (can be None).
    :vartype y_rf: :class:`np.ndarray` or None
    :ivar class_names: The names representing each class.
    :vartype class_names: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of ReceptiveFieldsReport.

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *X_rf* (``np.ndarray``) --
                The matrix of coordinates for each receptive field.
            *   *zhat_rf* (``np.ndarray``) --
                The softmax scores for the predictions on each receptive field.
            *   *yhat_rf* (``np.ndarray``) --
                The predictions for each receptive field.
            *   *y_rf* (``np.ndarray``) --
                The expected values for each receptive field (OPTIONAL).
            *   *class_names* (``list``) --
                The name representing each class (OPTIONAL). If not given,
                C0, ..., CN will be used by default.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.X_rf = kwargs.get('X_rf', None)
        if self.X_rf is None:
            raise ReportException(
                'Receptive field report is not possible without the '
                'coordinates for each receptive field.'
            )
        self.zhat_rf = kwargs.get('zhat_rf', None)
        self.yhat_rf = kwargs.get('yhat_rf', None)
        self.y_rf = kwargs.get('y_rf', None)
        self.class_names = kwargs.get('class_names', None)
        if self.class_names is None:
            self.class_names = [f'C{i}' for i in range(self.zhat_rf.shape[-1])]

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (receptive fields as point clouds) to files (LAZ).

        :param path: Path to the directory where the reports must be written.
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
            path,
            'Cannot find the directory to write the receptive fields:'
        )
        # Write each receptive field
        fnames = [f'softmax_{cname}' for cname in self.class_names]
        if self.yhat_rf is not None:
            fnames.append('Predictions')
        for i in range(len(self.X_rf)):
            path_rf = os.path.join(path, f'receptive_field_{i}.laz')
            PointCloudIO.write(
                PointCloudFactoryFacade.make_from_arrays(
                    self.X_rf[i],
                    np.hstack([
                        self.zhat_rf[i],
                        np.expand_dims(self.yhat_rf[i], -1)
                    ]) if self.yhat_rf is not None else self.zhat_rf,
                    self.y_rf[i],
                    fnames=fnames
                ),
                path_rf
            )
        # Log
        LOGGING.LOGGER.info(f'Receptive fields written to "{path}"')
