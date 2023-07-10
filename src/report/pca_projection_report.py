# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PCAProjectionReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to PCA projection.
    See :class:`.Report`.
    See also :class:`.VarianceSelector`.

    :ivar pca_names: The names of the PCA-derived features.
    :vartype pca_names: list
    :ivar expl_var_ratio: The vector which components represent the
        explained variance ratio.
    :vartype expl_var_ratio: :class:`np.ndarray`
    :ivar in_dim: The input dimensionality, i.e., the number of features
        before the PCA projection.
    :vartype in_dim: int
    :param kwargs: The key-word arguments.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, pca_names, expl_var_ratio, in_dim, **kwargs):
        """
        Initialize an instance of PCAProjectionReport.

        :param pca_names: The names of the PCA-derived features.
        :type pca_names: list
        :param expl_var_ratio: The vector which components represent the
            explained variance ratio.
        :type expl_var_ratio: :class:`np.ndarray`
        :param in_dim: The input dimensionality, i.e., the number of features
            before the PCA projection.
        :type in_dim: int
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the PCAProjectionReport
        self.pca_names = pca_names
        self.expl_var_ratio = expl_var_ratio
        self.in_dim = in_dim
        # Validate
        if self.expl_var_ratio is None:
            raise ReportException(
                'PCAProjectionReport is not possible without explained '
                'variances.'
            )
        if self.in_dim is None:
            raise ReportException(
                'PCAProjectionReport is not possible without knowledge '
                'on the input dimensionality.'
            )
        # Handle pca_names when they are None
        if self.pca_names is None:
            self.pca_names = [
                f'PCA_{i}' for i in range(1, len(self.expl_var_ratio))
            ]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the PCA projection report.
        See :class:`.Report` and also :meth:`report.Report.__str__`
        """
        # --- Introduction --- #
        s = f'PCA projection from {self.in_dim} to ' \
            f'{len(self.expl_var_ratio)} features:\n\n'
        # --- Table: feature, variance --- #
        s += 'FEATURE                  , EXPLAINED VAR. (%)\n'
        expl_var_sort = np.argsort(self.expl_var_ratio)
        for i in expl_var_sort:
            pca_name = self.pca_names[i]
            ev_ratio = self.expl_var_ratio[i]
            s += f'{pca_name:24.24} , {100*ev_ratio:18.4f}\n'
        # --- Total explained variance --- #
        s += '\nTOTAL EXPLAINED VARIANCE: ' \
            f'{100*np.sum(self.expl_var_ratio):.4f}%'
        # Return
        return s

