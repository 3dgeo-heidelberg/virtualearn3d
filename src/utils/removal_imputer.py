# ---   IMPORTS   --- #
# ------------------- #
from src.utils.imputer import Imputer, ImputerException
import numpy as np


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
        The fundamental imputation logic defining the removal imputer
        See :class:`.Imputer`
        """
        # Check
        if F is None:
            raise ImputerException(
                'Cannot remove NaN from data if there is no data.'
            )
        # Nan mask is true for nan and false for not nan
        if self.target_val.lower() == "nan":
            nan_mask = np.bitwise_or.reduce(np.isnan(F), axis=1)
        else:
            nan_mask = np.bitwise_or.reduce(F == self.target_val, axis=1)
        # Return
        if y is not None:
            return F[~nan_mask], y[~nan_mask]
        return F[~nan_mask]


