# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.main_config import VL3DCFG
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PredSelectStrategy:
    """
    :author: Alberto M. Esmoris Pena

    Interface for select operations on predictions.

    See :class:`.PredictionReducer`.
    """
    # ---   INIT   --- #
    # ⨪--------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a prediction selection strategy.

        :param kwargs: The attributes for the PredSelectStrategy.
        """
        self.prediction_data_type = np.uint64
        classification_bits = VL3DCFG['MODEL']['classification_space_bits']
        if classification_bits == 8:
            self.prediction_data_type = np.uint8
        elif classification_bits == 16:
            self.prediction_data_type = np.uint16
        elif classification_bits == 32:
            self.prediction_data_type = np.uint32

    # ---  SELECT METHODS  --- #
    # ------------------------ #
    @abstractmethod
    def select(self, reducer, Z):
        r"""
        The method that provides the logit to select the values of interest
        from the reduced predictions. It must be overridenn by any concrete
        implementation of a prediction selection strategy.

        :param reducer: The prediction reducer that is doing the reduction.
        :type reducer: :class:`.PredictionReducer`
        :param Z: Matrix-like array. There are as many rows as points and
            as many columns as reduced predictions.
        :return: The selected predictions derived from the reduced predictions
            as a matrix. Either a matrix with the :math:`n_y` point-wise output
            values :math:`\pmb{\hat{Y}} \in \mathbb{R}^{m \times n_y}` or a
            vector for the case of a single point-wise scalar output
            :math:`\pmb{\hat{y}} \in \mathbb{R}^{m}`.
        :rtype: :class:`np.ndarray`
        """
        pass
