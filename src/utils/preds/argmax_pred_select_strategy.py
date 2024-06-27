# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.pred_select_strategy import PredSelectStrategy
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ArgMaxPredSelectStrategy(PredSelectStrategy):
    r"""
    :author: Alberto M. Esmoris Pena

    Select the index of the max prediction from reduced predictions.

    The selected prediction for the :math:`i-th` points assuming
    :math:`K` predicted values (e.g., likelihoods for classifications) will be:

    .. math::
        y_{i} = \operatorname*{argmax}_{0 \leq k < K} \quad z_{ik}

    Note that when a single value is given the selection will consider the
    value round to the closest integer such that:

    .. math::
        y_{i} = \lfloor{z_i}\rceil

    See :class:`.PredSelectStrategy`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an argmax prediction selection strategy.

        :param kwargs: The attributes for the ArgmaxPredSelectStrategy.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---  SELECT METHODS  --- #
    # ------------------------ #
    def select(self, reducer, Z):
        """
        See :class:`.PredSelectStrategy` and
        :meth:`.PredSelectStrategy.select`.
        """
        # When there is only a single value, round it to the closest integer
        if ArgMaxPredSelectStrategy.is_single_value(Z):
            return ArgMaxPredSelectStrategy.round_to_closest_int(Z).astype(
                self.prediction_data_type
            )
        # Otherwise, take the argmax
        return np.argmax(Z, axis=1).astype(self.prediction_data_type)

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def is_single_value(Z):
        """
        Check whether the reduced predictions consist of a single scalar per
        point (True) or not (False).

        :param Z: The reduced predictions to be checked.
        :return: True if the reduced predictions consist of a single scalar,
            False otherwise.
        :rtype: bool
        """
        return len(Z.shape) <= 1

    @staticmethod
    def round_to_closest_int(Z):
        """
        Round each reduced prediction to its closest integer.

        :param Z: The reduced predictions to be checked.
        :return: Each prediction rounded to its closest integer.
        """
        return np.round(Z)
