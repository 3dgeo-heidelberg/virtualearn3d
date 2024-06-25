# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class DataAugmentor:
    r"""
    :author: Alberto M. Esmoris Pena

    Interface representing a data augmentation object. Any class must realize
    this interface if it aims to provide data augmentation capabilities.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        pass

    # ---  DATA AUGMENTATION METHODS  --- #
    # ----------------------------------- #
    @abstractmethod
    def augment(self, X, **kwargs):
        """
        Augment the given input data.

        :param X: The input data.
        :param kwargs: Key-word arguments for data augmentors that need
            further input specifications.
        :return: The augmented input data.
        """
        pass
