# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.regularizer.regularizer import Regularizer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesOrthogonalRegularizer(Regularizer):
    r"""
    FeaturesOrthogonalRegularizer regularizer to enforce orthogonality in the
    feature space.
    Taken from the Keras web documentation on 2023-07-19
    https://keras.io/examples/vision/pointnet
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the FeaturesOrthogonalRegularizer regularizer.
        See :meth:`regularizer.Regularizer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign orthogonal regularizer attributes
        self.num_features = kwargs.get('num_features', None)
        if self.num_features is None:
            raise DeepLearningException(
                'FeaturesOrthogonalRegularizer regularizer cannot be '
                'instantiated without the number of features. '
                'None was specified.'
            )
        self.l2reg = kwargs.get('l2reg', 0.001)
        self.eye = tf.eye(self.num_features)

    # ---   REGULARIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, x, training=False, mask=False):
        """
        The computational logic of the features orthogonal regularizer.
        See :meth:`regularizer.Regularizer.__call__`.
        :param x: The input tensor.
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        """
        Obtain the dictionary specifying how to serialize the features
        orthogonal regularizer.

        :return: The dictionary with the necessary data to serialize the
            features orthogonal regularizer.
        :rtype: dict
        """
        # Get dictionary from parent
        cfg = super().get_config()
        # Add state variables from features orthogonal regularizer
        cfg['num_features'] = self.num_features
        cfg['l2reg'] = self.l2reg
        # Return dictionary of state variables
        return cfg

    @classmethod
    def from_config(cls, cfg):
        """
        Deserialize a features ortoghonal regularizer from given specification.

        :param cfg: The dictionary specifying how to deserialize the
            regularizer.

        :return: The deserialized orthogonal regularizer.
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
