# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.regularizer.regularizer import Regularizer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesOrthogonalRegularizer(Regularizer):
    r"""
    FeaturesOrthogonalRegularizer layer to enforce orthogonality in the
    feature space.
    Taken from the Keras web documentation on 2023-07-19
    https://keras.io/examples/vision/pointnet
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the FeaturesOrthogonalRegularizer layer.
        See :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign orthogonal regularizer attributes
        self.num_features = kwargs.get('num_features', None)
        if self.num_features is None:
            raise DeepLearningException(
                'FeaturesOrthogonalRegularizer layer cannot be instantiated '
                'without the number of features. None was specified.'
            )
        self.l2reg = kwargs.get('l2reg', 0.001)
        self.eye = tf.eye(self.num_features)

    # ---   REGULARIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, x, training=False, mask=False):
        """
        The computational logic of the orthogonal regularizer.
        See :meth:`layer.Layer.call`.
        :param x: The input tensor.
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        """
        Obtain the dictionary specifying how to serialize the orthogonal
        regularizer layer.

        :return: The dictionarty with the necessary data to serialize the
            orthogonal regularizer layer.
        :rtype: dict
        """
        # Return dictionary of state variables
        return {
            'num_features': self.num_features,
            'l2reg': self.l2reg
        }

    @classmethod
    def from_config(cls, cfg):
        """
        Deserialize an ortoghonal regularizer layer from given specification.

        :param cfg: The dictionary specifying how to deserialize the layer.

        :return: The deserialized orthogonal regularizer layer.
        :rtype: :class:.FeaturesOrthogonalRegularizer`
        """
        # Instantiate from cfg
        ortho_reg = cls(**cfg)
        # Assign member attributes
        ortho_reg.num_features = cfg['num_features']
        ortho_reg.l2reg = cfg['l2reg']
        ortho_reg.eye = tf.eye(ortho_reg.num_features)
        # Return
        return ortho_reg
