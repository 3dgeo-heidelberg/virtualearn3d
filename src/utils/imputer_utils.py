# ---   IMPORTS   --- #
# ------------------- #
from src.utils.removal_imputer import RemovalImputer
from src.utils.univariate_imputer import UnivariateImputer


# ---   CLASS   --- #
# ----------------- #
class ImputerUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with imputers.
    """

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_imputer_class(spec):
        """
        Extract the imputer's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing an imputer.
        :rtype: :class:`.Imputer`
        """
        imputer = spec.get('imputer', None)
        if imputer is None:
            raise ValueError(
                'Imputing a dataset requires an imputer. None was specified.'
            )
        # Check imputer class
        imputer_low = imputer.lower()
        if imputer_low == 'removalimputer':
            return RemovalImputer
        if imputer_low == 'univariateimputer':
            return UnivariateImputer
        # An unknown imputer was specified
        raise ValueError(f'There is no known imputer "{imputer}"')
