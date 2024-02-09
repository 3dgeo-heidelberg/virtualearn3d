# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class KPConvLayer(Layer):
    """

    :ivar sigma: The influence distance governing the kernel.
    :vartype sigma: float
    """
    # TODO Rethink : Doc
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        sigma=1.0,
        kernel_radius=1.0,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.sigma = sigma
        self.kernel_radius = kernel_radius
        # TODO Rethink : Implement

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        # TODO Rethink : Doc
        pass  # TODO Rethink : Implement

    def call(self, inputs, training=False, mask=False):
        # Extract input
        X = inputs[0]
        F = inputs[1]
        N = inputs[2]
        # TODO Rethink : Implement

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    # TODO Rethink : Implement serialization
