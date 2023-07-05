# ---   IMPORTS   --- #
# ------------------- #
from src.utils.imput.imputer import Imputer
from sklearn.impute import SimpleImputer
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class UnivariateImputer(Imputer):
    """
    :author: Alberto M. Esmoris Pena

    Class to compute univariate imputations.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a UnivariateImputer

        :param kwargs: The attributes for the UnivariateImputer
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes for the UnivariateImputer
        target_val = self.target_val
        if target_val.lower() == "nan":
            target_val = np.nan
        self.imputer = SimpleImputer(
            missing_values=target_val,
            strategy=kwargs.get('strategy', 'mean'),
            fill_value=kwargs.get('constant_val', 0)
        )

    # ---   IMPUTER METHODS   --- #
    # --------------------------- #
    def impute(self, F, y=None):
        """
        The fundamental imputation logic defining the univariate imputer
        See :class:`.Imputer`

        In this case, since imputation will not remove points, the y argument
        will be ignored, no matter what. However, in case y is given as not
        None, the return will be (imputed F, y) for compatibility and fluent
        programming. If y is None, only imputed F will be return.
        """
        if y is not None:
            return self.imputer.fit_transform(F), y
        return self.imputer.fit_transform(F)
