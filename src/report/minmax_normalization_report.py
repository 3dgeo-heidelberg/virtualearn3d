# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class MinmaxNormalizationReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to MinmaxNormalization.
    See :class:`.Report`.
    See also :class:`.MinmaxNormalizer`.

    :ivar fnames: The names of the features.
    :vartype fnames: list
    :ivar fmin: The feature-wise min values.
    :vartype fmin: :class:`np.ndarray`
    :ivar fmax: The feature-wise max values.
    :vartype fmax: :class:`np.ndarray`
    :ivar frange: The feature-wise range values, i.e., max-min.
    :vartype frange: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fnames, fmin, fmax, **kwargs):
        """
        Initialize an instance of MinmaxNormalizationReport

        :param fnames: The names of the features.
        :type fnames: list
        :param fmin: The feature-wise min values.
        :type fmin: :class:`np.ndarray`
        :param fmax: The feature-wise max values.
        :type fmax: :class:`np.ndarray`
        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *range* (``np.ndarray``) --
                The feature-wise range values, i.e., max-min.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the MinmaxNormalizationReport
        self.fnames = fnames
        self.fmin = fmin
        self.fmax = fmax
        self.frange = kwargs.get('frange', None)
        # Validate
        if self.fmin is None:
            raise ReportException(
                'MinmaxNormalizationReport requires min values. '
                'None were given.'
            )
        if self.fmax is None:
            raise ReportException(
                'MinmaxNormalizationReport requires max values. '
                'None were given.'
            )
        # If feature range is not given, derive it from min and max
        if self.frange is None:
            self.frange = self.fmax - self.fmin

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the minmax normalization report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = f'Min-max normalization of {len(self.fnames)} features:\n\n'
        # --- Table: feature, min, max, range --- #
        # HEAD
        s += 'FEATURE         ,              MIN,              MAX, '\
            '           RANGE\n'
        # BODY
        for i, fname in enumerate(self.fnames):
            s += f'{fname:24.24}, {self.fmin[i]:16.5f}, '\
                f'{self.fmax[i]:16.5f}, {self.frange[i]:16.5f}\n'
        # Return
        return s
