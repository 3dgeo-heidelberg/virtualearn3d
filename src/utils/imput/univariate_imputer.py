# ---   IMPORTS   --- #
# ------------------- #
from src.utils.imput.imputer import Imputer
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from sklearn.impute import SimpleImputer
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class UnivariateImputer(Imputer):
    """
    :author: Alberto M. Esmoris Pena

    Class to compute univariate imputations.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_imputer_args(spec):
        """
        Extract the arguments to initialize/instantiate an UnivariateImputer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an UnivariateImputer.
        """
        # Initialize from parent
        kwargs = Imputer.extract_imputer_args(spec)
        # Extract particular arguments of UnivariateImputer
        kwargs['strategy'] = spec.get('strategy', None)
        kwargs['constant_val'] = spec.get('constant_val', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

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
        self.fit = False

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
        # Report feature names
        LOGGING.LOGGER.debug(
            f'UnivariateImputer considers the following {len(self.fnames)} '
            f'features:\n{self.fnames}'
        )
        # Fit imputer
        if not self.fit:
            start = time.perf_counter()
            self.imputer.fit(F)
            end = time.perf_counter()
            self.fit = True
            LOGGING.LOGGER.info(
                f'UnivariateImputer fit to {F.shape[0]} points with '
                f'{F.shape[1]} features in {end-start:.3f} seconds.'
            )
        # Impute data
        start = time.perf_counter()
        F = self.imputer.transform(F)
       end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'UnivariateImputer imputed {F.shape[0]} points with {F.shape[1]} '
            f'features in {end-start:.3f} seconds.'
        )
        # Return
        if y is not None:
            return F, y
        return F
