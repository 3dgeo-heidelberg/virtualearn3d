# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class Initializer(tf.keras.initializers.Initializer):
    r"""
    :author: Alberto M. Esmoris Pena

    An initializer can be seen as a routine that initializes the internal
    state of a layer from a neural network. Typically, initializers receive
    the shape of the expected output tensor as a parameter.

    The Initializer class provides an interface that must be realized by any
    class that must assume the role of a regularizer inside a neural network.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the member attributes of the initializer.

        :param kwargs: The key-word specification to parametrize the
            initializer.
        """
        # Call parent's init
        super().__init__()

    # ---   INITIALIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, shape, dtype=None):
        """
        The initializer's computation logic.

        :param shape: The shape of the variable to initializer.
        :param dtype: The type of value.
        :return: The initialized variable.
        """
        raise DeepLearningException(
            "Initializer has no __call__ method. Derived classes must "
            "overload the __call__ method to define the computational "
            "logic of the layer."
        )

    def get_config(self):
        """
        The dictionary specifying how to serialize the initializer.

        :return: The dictionary with the necessary data to serialize the
            initializer.
        :rtype: dict
        """
        return {}
