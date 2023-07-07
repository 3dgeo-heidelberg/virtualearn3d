# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class VarianceSelectionReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to variance selection.
    See :class:`.Report`.
    See also :class:`.VarianceSelector`.

    :ivar fnames: The names of all the features.
    :vartype fnames: list
    :ivar variances: The feature-wise variances.
    :vartype variances: :class:`np.ndarray`
    :ivar selected_features: The indices or bool mask representing the selected
        features. This is an optional attribute, if it is not available, then
        selected features will not be reported.
    :vartype selected_features: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fnames, variances, selected_features=None, **kwargs):
        """
        Initialize an instance of VarianceSelectionReport.

        :param fnames: The names of all the features (OPTIONAL). If not given,
            the default will be f1, ..., fn.
        :param variances: The feature-wise variances.
        :param selected_features: The indices or bool mask representing the
            selected features (OPTIONAL).
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the VarianceSelectionReport
        self.fnames = fnames
        self.variances = variances
        self.selected_features = selected_features
        # Validate
        if self.variances is None:
            raise ReportException(
                'VarianceSelectionReport is not possible without variances.'
            )
        # Handle fnames when they are None
        if self.fnames is None:
            self.fnames = [f'f{i}' for i in range(1, len(self.variances)+1)]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the variance selection report.
        See :class:`.Report` and also :meth:`report.Report.__str__`
        """
        # --- Introduction --- #
        s = f'Variance selection on {len(self.variances)} features'
        if self.selected_features is not None:
            if self.selected_features.dtype == bool:
                s += f', selecting {np.count_nonzero(self.selected_features)}'
            else:
                s += f', selecting {len(self.selected_features)}'
        s += ':\n\n'
        # --- Table: feature, variance --- #
        s += 'FEATURE                  , VARIANCE\n'
        var_sort = np.argsort(self.variances)
        for i in var_sort:
            fname = self.fnames[i]
            variance = self.variances[i]
            s += f'{fname:24.24} , {variance:12.3f}\n'
        # ---  Selected features  --- #
        if self.selected_features is not None:
            s += '\n\nSELECTED FEATURES:\n'
            for fname in np.array(self.fnames)[self.selected_features]:
                s += f'{fname}\n'
        # Return
        return s
