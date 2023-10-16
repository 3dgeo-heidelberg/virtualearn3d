# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ctransf.class_reducer import ClassReducer


# ---   CLASS   --- #
# ----------------- #
class CtransfUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with class transformers.
    """

    # ---   EXTRACT FROM SPEC   --- #
    # ----------------------------- #
    @staticmethod
    def extract_ctransf_class(spec):
        """
        Extract the classification transformer's class from the key-word
        specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a class transformer.
        :rtype: :class:`.ClassTransformer`
        """
        ctransf = spec.get('class_transformer', None)
        if ctransf is None:
            raise ValueError(
                'Transforming classes requires a class transformer. None '
                'was specified.'
            )
        # Check class transformer class
        ctransf_low = ctransf.lower()
        if ctransf_low == 'classreducer':
            return ClassReducer
        # An unknown class transformer was specified
        raise ValueError(f'There is no known class transformer "{ctransf}"')
