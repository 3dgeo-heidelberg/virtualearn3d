# ---   IMPORTS   --- #
# ------------------- #
from src.utils.imput.imputer import Imputer, ImputerException
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class RemovalImputer(Imputer):
    """
    :author: Alberto M. Esmoris Pena

    Class to remove missing values.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a RemovalImputer

        :param kwargs: The attributes for the RemovalImputer
        """
        # Call parent init
        super().__init__(**kwargs)

    # ---   IMPUTER METHODS   --- #
    # --------------------------- #
    def impute(self, F, y=None):
        """
        The fundamental imputation logic defining the removal imputer.

        See :class:`.Imputer` and :meth:`imputer.Imputer.impute`.
        """
        # Report feature names
        LOGGING.LOGGER.debug(
            f'RemovalImputer considers the following {len(self.fnames)} '
            f'features:\n{self.fnames}'
        )
        start = time.perf_counter()
        # Check
        if F is None:
            raise ImputerException(
                'Cannot remove NaN from data if there is no data.'
            )
        # The mask is True when the target value is matched, False otherwise
        # e.g., nan mask is true for nan and false for not nan
        mask = self.find_target_mask(F)
        # Log imputation
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'RemovalImputer removed {np.count_nonzero(mask)} points with '
            f'missing values "{self.target_val}" in {end-start:.3f} seconds.'
        )
        # Return
        if y is not None:
            return F[~mask], y[~mask]
        return F[~mask]

    def impute_pcloud(self, pcloud, fnames=None):
        """
        Overwrite the logic of :meth:`.Imputer.impute_pcloud` because
        :class:`.RemovalImputer` might need to remove points from the point
        cloud.

        See :meth:`.Imputer.imput_pcloud`.
        """
        # Report feature names
        fnames = self.find_fnames(fnames=fnames)
        LOGGING.LOGGER.debug(
            f'RemovalImputer considers the following {len(fnames)} '
            f'features:\n{fnames}'
        )
        start = time.perf_counter()
        # Obtain features matrix
        P = self.extract_pcloud_matrix(pcloud, fnames)  # Might include coords.
        # Obtain mask from features matrix
        mask = self.find_target_mask(P)
        # Remove points that match the target value
        pcloud = pcloud.remove_mask(mask)
        # Log imputation
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'RemovalImputer removed {np.count_nonzero(mask)} points with '
            f'missing values "{self.target_val}" in {end-start:.3f} seconds.'
        )
        # Return
        return pcloud

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def find_target_mask(self, F):
        """
        Obtain a boolean mask that specifies whether a given point matches the
        target value (True) or not (False).

        :param F: The matrix of point-wise features representing the point
            cloud.
        :return: The
        """
        if self.target_val.lower() == "nan":
            mask = np.bitwise_or.reduce(np.isnan(F), axis=1)
        else:
            mask = np.bitwise_or.reduce(F == self.target_val, axis=1)
        return mask
