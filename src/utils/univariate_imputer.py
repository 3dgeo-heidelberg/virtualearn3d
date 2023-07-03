# ---   IMPORTS   --- #
# ------------------- #
from src.utils.imputer import Imputer, ImputerException
from sklearn.impute import SimpleImputer


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
        self.imputer = SimpleImputer(
            missing_values=self.target_val,
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
        will be ignored, no matter what.
        """
        return self.imputer.fit_transform(F)
