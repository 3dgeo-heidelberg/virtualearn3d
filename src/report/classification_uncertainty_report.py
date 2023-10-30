# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
import numpy as np
import os


class ClassificationUncertaintyReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to the uncertainty of classified point
    clouds.

    See :class:`.Report`.
    :ivar X: See :class:`.ClassificationUncertaintyEvaluation`
    :ivar y: See :class:`.ClassificationUncertaintyEvaluation`
    :ivar yhat: See :class:`.ClassificationUncertaintyEvaluation`
    :ivar Zhat: See :class:`.ClassificationUncertaintyEvaluation`
    :ivar pwise_entropy: See :class:`.ClassificationUncertaintyEvaluation`
    :ivar weighted_hspace_entropy: See
        :class:`.ClassificationUncertaintyEvaluation`
    :ivar class_ambiguity: See :class:`.ClassificationUncertaintyEvaluation`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of ClassificationUncertaintyReport.

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *class_names* (``list``) --
                The name for each class.
            *   *X* (``np.ndarray``) --
                The matrix of coordinates representing the point cloud.
            *   *y* (``np.ndarray``) --
                The vector of point-wise classes (reference).
            *   *yhat* (``np.ndarray``) --
                The vector of predicted point-wise classes.
            *   *Zhat* (``np.ndarray``) --
                The matrix of point-wise probabilities corresponding to the
                predicted classes.
            *   *pwise_entropy* (``np.ndarray``) --
                The vector of point-wise Shannon's entropy.
            *   *weighted_hspace_entropy* (``np.ndarray``) --
                The vector of point-wise weighted Shannon's entropy on features
                half-spaces.
            *   *class_ambiguity* (``np.ndarray``) --
                The vector of point-wise class ambiguities.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.class_names = kwargs['class_names']
        self.X = kwargs['X']
        self.y = kwargs.get('y', None)
        self.yhat = kwargs.get('yhat', None)
        self.Zhat = kwargs.get('Zhat', None)
        self.pwise_entropy = kwargs.get('pwise_entropy', None)
        self.weighted_hspace_entropy = kwargs.get(
            'weighted_hspace_entropy', None
        )
        self.class_ambiguity = kwargs.get('class_ambiguity', None)

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (point cloud) to a file (LAS/LAZ).

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
            'Cannot find the directory to write the evaluation of the '
            'classification\'s uncertainties.'
        )
        # Build output
        onames = ['Prediction']
        outs = self.yhat.reshape((-1, 1))
        # Build output : success mask
        if self.y is not None:
            onames.append('Success')
            success = (self.y == self.yhat).astype(int).reshape((-1, 1))
            outs = np.hstack([outs, success])
        # Build output : class-wise probabilities
        if self.Zhat is not None:
            if len(self.Zhat.shape) == 1:  # Handle binary classif case
                onames = onames + [
                    f'{self.class_names[0]}_to_{self.class_names[1]}'
                ]
            else:  # Otherwise, general case
                onames = onames + self.class_names
            outs = np.hstack([
                outs,
                self.Zhat if len(self.Zhat.shape) > 1
                else self.Zhat.reshape((-1, 1))
            ])
        # Build output :  Point-wise entropy
        if self.pwise_entropy is not None:
            onames.append('PwiseEntropy')
            outs = np.hstack([
                outs,
                self.pwise_entropy.reshape((-1, 1))
            ])
        # Build output : Weighted half-space entropy
        if self.weighted_hspace_entropy is not None:
            onames.append('WHspaceEntropy')
            outs = np.hstack([
                outs,
                self.weighted_hspace_entropy.reshape((-1, 1))
            ])
        # Build output : Class ambiguity
        if self.class_ambiguity is not None:
            onames.append('ClassAmbiguity')
            outs = np.hstack([
                outs,
                self.class_ambiguity.reshape((-1, 1))
            ])
        # Write
        PointCloudIO.write(
            PointCloudFactoryFacade.make_from_arrays(
                self.X,
                outs,
                self.y,
                fnames=onames
            ),
            path
        )
        # Log
        LOGGING.LOGGER.info(
            f'Classification uncertainty point cloud written to "{path}"'
        )
