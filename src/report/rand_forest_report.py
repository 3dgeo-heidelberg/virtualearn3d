# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class RandForestReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to trained random forest models.
    See :class:`.Report`.
    See also :class:`.RandomForestClassificationModel`.

    :ivar importance: The vector of feature-wise importance.
    :vartype importance: :class:`np.ndarray`
    :ivar permutation_importance_mean: The vector of feature-wise mean
        permutation importance. It can be None, i.e., it is optional.
    :vartype permutation_importance_mean: :class:`np.ndarray`
    :ivar permutation_importance_stdev: The vector representing the standard
        deviations of permutation importance. It can be None, i.e., it is
        optional.
    :vartype permutation_importance_stdev: :class:`np.ndarray`
    :ivar fnames: The names of the features.
    :vartype fnames: list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, importance, **kwargs):
        """
        Initialize an instance of RandForestReport.

        :param kwargs: The key-word arguments defining the report's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the RandForestReport
        self.importance = importance
        self.permutation_importance_mean = kwargs.get(
            'permutation_importance_mean', None
        )
        self.permutation_importance_stdev = kwargs.get(
            'permutation_importance_stdev', None
        )
        self.fnames = kwargs.get('fnames', None)
        # Handle serial fnames when None are given
        if self.fnames is None:
            self.fnames = [f'f_{i}' for i in range(len(self.importance))]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the report about a trained random forest.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = 'Random forest and importance from training features:\n'
        # --- Head --- #
        s += 'FEATURE                 , FEATURE IMPORTANCE  '
        if self.permutation_importance_mean is not None:
            s += ', PERM. IMP. MEAN     , PERM. IMP. STDEV'
        # --- Body --- #
        sorted_indices = np.argsort(self.importance)
        for i in sorted_indices:
            s += f'\n{self.fnames[i]:24.24}, {self.importance[i]:20.6f}'
            if self.permutation_importance_mean is not None:
                s += f', {self.permutation_importance_mean[i]:20.6f}, '\
                    f'{self.permutation_importance_stdev[i]:20.6f}'
        # Return
        return s
