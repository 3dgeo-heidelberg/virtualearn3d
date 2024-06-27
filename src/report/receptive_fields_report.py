# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
from src.main.main_config import VL3DCFG
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
    :ivar F_rf: The matrix of features for each receptive field.
    :vartype F_rf: :class:`np.ndarray`
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
            *   *F_rf* (``np.ndarray``) --
                The matrix of features for each receptive field.
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
        self.F_rf = kwargs.get('F_rf', None)
        self.fnames = kwargs.get('fnames', None)
        if self.fnames is None and self.F_rf is not None:
            # Derive feature names automatically
            self.fnames = [f'f{i+1}' for i in range(self.F_rf[0].shape[-1])]
        if (
            self.F_rf is not None and self.fnames is not None and
            self.F_rf[0].shape[-1] != len(self.fnames)
        ):  # Validate feature names match given features
            raise ReportException(
                f'Receptive field report received {len(self.fnames)} names '
                f'for {len(self.F_rf[0])} features. However, the number of '
                'features MUST match the number of feature names.'
            )
        self.zhat_rf = kwargs.get('zhat_rf', None)
        self.yhat_rf = kwargs.get('yhat_rf', None)
        self.y_rf = kwargs.get('y_rf', None)
        self.class_names = kwargs.get('class_names', None)
        if self.class_names is None:
            self.class_names = [f'C{i}' for i in range(self.zhat_rf.shape[-1])]
        # Attributes from config
        CFG = VL3DCFG['REPORT']['ReceptiveFieldsReport']
        self.include_entropies = CFG.get('include_entropies', True)
        self.include_likelihoods = CFG.get('include_likelihoods', True)
        self.include_predictions = CFG.get('include_predictions', True)
        self.include_references = CFG.get('include_references', True)
        self.include_success = CFG.get('include_success', True)
        self.include_features = CFG.get('include_features', False)

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
        fnames = []
        if self.F_rf is not None and self.include_features:
            fnames.extend(self.fnames)
        if self.zhat_rf is not None and self.include_likelihoods:
            if self.zhat_rf.shape[-1] == 1:  # Assume binary classification
                fnames.extend([
                    f'{self.class_names[0]}_to_{self.class_names[1]}'
                ])
            else:
                fnames.extend([
                    f'softmax_{cname}' for cname in self.class_names
                ])
        if self.yhat_rf is not None and self.include_predictions:
            fnames.append('Predictions')
        if(
            self.y_rf is not None and self.yhat_rf is not None and
            self.include_success
        ):
            fnames.append('Success')
        if self.zhat_rf is not None and self.include_entropies:
            fnames.append('PointWiseEntropy')
        for i in range(len(self.X_rf)):
            path_rf = os.path.join(path, f'receptive_field_{i}.laz')
            F_rfi = []
            if self.F_rf is not None and self.include_features:
                F_rfi.append(self.F_rf[i])
            if self.zhat_rf is not None and self.include_likelihoods:
                F_rfi.append(self.zhat_rf[i])
            if self.yhat_rf is not None and self.include_predictions:
                F_rfi.append(np.expand_dims(self.yhat_rf[i], -1))
            if(
                self.y_rf is not None and self.yhat_rf is not None and
                self.include_success
            ):
                F_rfi.append(np.expand_dims(
                    (self.y_rf[i] == self.yhat_rf[i]).astype(self.y_rf[i].dtype),
                    -1
                ))
            if self.zhat_rf is not None and self.include_entropies:
                pwe_i = -np.log2(self.zhat_rf[i]) * self.zhat_rf[i]
                if len(pwe_i.shape) > 1:
                    pwe_i = np.sum(pwe_i, axis=1)
                F_rfi.append(np.expand_dims(pwe_i, -1))
            PointCloudIO.write(
                PointCloudFactoryFacade.make_from_arrays(
                    self.X_rf[i],
                    np.hstack(F_rfi),
                    self.y_rf[i] if self.y_rf is not None else None,
                    fnames=fnames
                ),
                path_rf
            )
        # Log
        LOGGING.LOGGER.info(f'Receptive fields written to "{path}"')
