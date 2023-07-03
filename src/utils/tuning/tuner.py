# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException


# ---   EXCEPTIONS   --- #
# ---------------------- #
class TunerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to tuning components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Tuner:
    """
    :author: Alberto M. Esmoris Pena

    Class for model tuning operations.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a Tuner.

        :param kwargs: The attributes for the Tuner.
        """
        # Fundamental initialization of any tuner
        pass

    # ---   TUNER METHODS   --- #
    # ------------------------- #
    @abstractmethod
    def tune(self, model, pcloud=None):
        """
        Tune the given model on givel point cloud (if any).

        :param model: The model to be tuned. See :class:`.Model`
        :param pcloud: The (OPTIONAL) point cloud involved in the tuning.
            See :class:`.PointCloud`
        :return: The tuned model.
        :rtype: :class:`.Model`
        """
        pass
