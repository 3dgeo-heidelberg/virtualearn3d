# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException


# ---   EXCEPTIONS   --- #
# ---------------------- #
class DeepLearningException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to deep learning components.
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)
