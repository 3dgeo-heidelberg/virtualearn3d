# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.variance_selector import VarianceSelector
from src.utils.ftransf.kbest_selector import KBestSelector
from src.utils.ftransf.percentile_selector import PercentileSelector


# ---   CLASS   --- #
# ----------------- #
class FtransfUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with feature transformers.
    """

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_ftransf_class(spec):
        """
        Extract the feature transformer's class from the key-word
        specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a feature transformer.
        :rtype: :class:`.FeatureTransformer`
        """
        ftransf = spec.get('feature_transformer', None)
        if ftransf is None:
            raise ValueError(
                'Transforming features requires a feature transformer. None '
                'was specified.'
            )
        # Check feature transformer class
        ftransf_low = ftransf.lower()
        if ftransf_low == 'varianceselector':
            return VarianceSelector
        if ftransf_low == 'kbestselector':
            return KBestSelector
        if ftransf_low == 'percentileselector':
            return PercentileSelector
        # An unknown feature transformer was specified
        raise ValueError(f'There is no known feature transformer "{ftransf}"')
