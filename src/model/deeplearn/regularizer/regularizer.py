# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class Regularizer(tf.keras.regularizers.Regularizer):
    r"""
    :author: Alberto M. Esmoris Pena

    A regularizer can be seen as a map :math:`f` from an input tensor
    :math:`\mathcal{X}` to an output scalar :math:`y` that can be added to the
    loss function.

    The Regularizer class provides an interface that must be realized by any
    class that must assume the role of a regularizer inside a neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the member attributes of the regularizer.

        :param kwargs: The key-word specification to parametrize the
            regularizer.
        """
        # Call parent's init
        super().__init__()

    # ---   REGULARIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, x):
        """
        The layer's computation logic.

        :param x: The input tensor.
        :return: The output scalar.
        """
        raise DeepLearningException(
            "Regularizer has no __call__ method. Derived classes must "
            "overload the __call__ method to define the computational "
            "logic of the layer."
        )
