# ---   IMPORTS   --- #
# ------------------- #
from src.utils.tuning.hyper_grid_search import HyperGridSearch
from src.utils.tuning.hyper_random_search import HyperRandomSearch


# ---   CLASS   --- #
# ----------------- #
class TunerUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with model tuners.
    """

    # ---   EXTRACT FROM SPEC   --- #
    # ----------------------------- #
    @staticmethod
    def extract_tuner_class(spec):
        """
        Extract the tuner's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a tuner.
        :rtype: :class:`.Tuner`
        """
        tuner = spec.get('tuner', None)
        if tuner is None:
            raise ValueError(
                'Tuning a model requires a tuner. None was specified.'
            )
        # Check tuner class
        tuner_low = tuner.lower()
        if tuner_low == 'gridsearch':
            return HyperGridSearch
        if tuner_low == 'randomsearch':
            return HyperRandomSearch
        # An unknown tuner was specified
        raise ValueError(f'There is no known tuner "{tuner}"')
