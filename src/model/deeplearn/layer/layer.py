# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class Layer(tf.keras.layers.Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A layer can be seen as a map :math:`f` from an input tensor
    :math:`\mathcal{X}` to an output tensor :math:`\mathcal{Y}`, i.e.,
    :math:`f(\mathcal{X}) = \mathcal{Y}`.

    The Layer class provides an interface that must be realized by any class
    that must assume the role of a layer inside a neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the member attributes of the layer and the internal weights
        that do not depend on the input dimensionality.

        :param kwargs: The key-word specification to parametrize the layer.
        """
        # Call parent's init
        super().__init__()

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        Logic to build the layer before the first call is executed.

        This method can be overloaded by any derived class to either change
        or extend the logic.

        :param dim_in: The dimensionality of the input tensor.
        """
        # Call parent's build
        super().build(dim_in)

    def call(self, inputs, training=False, mask=False):
        """
        The layer's computation logic.

        :param inputs: The input tensor or the list of input tensors.
        :param training: True when the layer is called during training, False
            otherwise.
        :type training: bool
        :param mask: Boolean mask (one boolean per input timestep).
        :return: The output tensor.
        """
        raise DeepLearningException(
            "Layer has no call method. Derived classes must overload the call "
            "method to define the computational logic of the layer."
        )

    def get_config(self):
        """
        Obtain the dictionary specifying how to serialize the layer.

        :return: The dictionary with the necessary data to serialize the layer.
        :rtype: dict
        """
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        """
        Deserialize a layer from given specification.

        :param config: The dictionary specifying how to deserialize the layer.
        :return: The deserialized layer.
        :rtype: :class:`.Layer` or derived
        """
        return cls(**config)
