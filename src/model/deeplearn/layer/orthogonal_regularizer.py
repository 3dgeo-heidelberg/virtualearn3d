# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class OrthogonalRegularizer(Layer):
    r"""
    OrthogonalRegularizer layer to enforce orthogonality in the feautre space.
    Taken from the Keras web documentation on 2023-07-19
    https://keras.io/examples/vision/pointnet
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the OrthogonalRegularizer layer.
        See :meth:`layer.Layer.__init__`.
        """
        self.num_features = kwargs.get('num_features', None)
        if self.num_features is None:
            raise DeepLearningException(
                'OrthogonalRegularizer layer cannot be instantiated without '
                'the number of features. None was specified.'
            )
        self.l2reg = kwargs.get('l2reg', None)
        self.eye = tf.eye(self.num_features)

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def call(self, x, training=False, mask=False):
        """
        The computational logic of the orthogonal regularizer.
        See :meth:`layer.Layer.call`.
        :param x: The input tensor.
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
