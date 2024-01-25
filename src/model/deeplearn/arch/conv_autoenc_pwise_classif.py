# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_post_processor import \
    HierarchicalFPSPostProcessor
import tensorflow as tf
# TODO Rethink : Implement

# ---   CLASS   --- #
# ----------------- #
class ConvAutoencPwiseClassif(Architecture):
    """
    :author: Alberto M. Esmoris Pena

    The convolutional autoencoder architecture for point-wise classification.

    Examples of convolutional autoencoders are the PointNet++ model
    (https://arxiv.org/abs/1706.02413) and the KPConv model
    (https://arxiv.org/abs/1904.08889).
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:architecture.Architecture.__init__`.
        :param kwargs:
        """