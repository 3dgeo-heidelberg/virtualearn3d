# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.layer.kpconv_layer import KPConvLayer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class StridedKPConvLayer(KPConvLayer):
    """

    :ivar sigma: The influence distance governing the kernel.
    :vartype sigma: float
    """
    # TODO Rethink : Doc
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        Also, see :class:`.KPConvLayer` and
        :meth:`kpconv_layer.KPConvLayer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        # Call parent's build
        super().build(dim_in)

    def call(self, inputs, training=False, mask=False):
        # Extract input
        Xa = inputs[0]
        Xb = inputs[1]
        Fa = inputs[2]
        ND = inputs[3]
        # TODO Rethink : Implement

