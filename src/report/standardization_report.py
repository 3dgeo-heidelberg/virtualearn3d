# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class StandardizationReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to Standardization report.
    See :class:`.Report`.
    See also :class:`.Standardizer`.

    :ivar fnames: The names of the features.
    :vartype fnames: list
    :ivar sigma: The vector of feature-wise standard deviations.
    :vartype sigma: :class:`np.ndarray`
    :ivar mu: The vector of feature-wise means.
    :vartype mu: :class:`np.ndarray
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fnames, sigma, mu, **kwargs):
        """
        Initialize an instance of StandardizationReport.

        :param fnames: The names of the features.
        :type fnames: list
        :param sigma: The vector of feature-wise standard deviations.
        :type sigma: :class:`np.ndarray`
        :param mu: The vector of feature-wise means.
        :type mu: :class:`np.ndarray`
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the StandardizationReport
        self.fnames = fnames
        self.sigma = sigma
        self.mu = mu
        # Validate
        if self.sigma is None and self.mu is None:
            raise ReportException(
                'StandardizationReport needs at least deviations (sigma) or '
                'means (mu). None were given.'
            )
        if self.sigma is not None and self.mu is not None:
            if self.sigma.shape[0] != self.mu.shape[0]:
                raise ReportException(
                    'StandardizationReport does not support a distinct number '
                    'of deviations (sigma) and means (mu).\n'
                    f'{self.sigma.shape[0]} deviations and {self.mu.shape[0]} '
                    'means were given.'
                )
        # Handle feature names when they are None
        if self.fnames is None:
            nfeats = self.sigma.shape[0] if self.sigma is not None else \
                self.mu.shape[0]
            self.fnames = [f'f{i}' for i in range(1, nfeats+1)]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the Standardization report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = f'Standardization of {len(self.fnames)} features:\n\n'
        # --- Table: feature, mean, stdev --- #
        # HEAD
        s += 'FEATURE                 , '
        if self.mu is not None:
            s += '            MEAN, '
        if self.sigma is not None:
            s += '          STDEV., '
        s = s[:-1] + '\n'  # Replace last space by a new line
        # BODY
        for i, fname in enumerate(self.fnames):
            s += f'{fname:24.24}'
            if self.mu is not None:
                s += f', {self.mu[i]:16.5f}'
            if self.sigma is not None:
                s += f', {self.sigma[i]:16.5f}'
            s += '\n'  # Add new line
        # Return
        return s
