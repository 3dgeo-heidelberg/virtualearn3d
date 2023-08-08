# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class PointNetPwiseClassif(PointNet):
    """
    :author: Alberto M. Esmoris Pena

    A specialization of the PointNet architecture for point-wise
    classification.

    See :class:`.PointNet`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:`architecture.PointNet.__init__`.
        """
        # Call parent's init
        kwargs['arch_name'] = 'PointNet_PointWise_Classification'
        super().__init__(**kwargs)
        # Assign the attributes of the PointNetPwiseClassif architecture
        self.num_classes = kwargs.get('num_classes', None)
        if self.num_classes is None:
            raise DeepLearningException(
                'The PointNetPwiseClassif architecture instantiation requires '
                'the number of classes defining the problem. None was given.'
            )
        self.num_pwise_feats = kwargs.get('num_pwise_feats', 128)
        self.binary_crossentropy = False
        comp_args = kwargs.get('compilation_args', None)
        if comp_args is not None:
            loss_args = comp_args.get('loss', None)
            if loss_args is not None:
                self.binary_crossentropy = \
                    loss_args.get('function', '') == 'binary_crossentropy'

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the PointNet neural network for point-wise
        classification tasks.

        See :meth:`point_net.PointNet.build_hidden`.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer.
        :rtype: :class:`tf.Tensor`.
        """
        # Call parent's build hidden
        x = super().build_hidden(x, **kwargs)
        # Extend parent's hidden layer with point-wise blocks
        x = tf.keras.layers.MaxPool1D(
            pool_size=self.num_points,
            name='max_pool1D'
        )(x)
        x = tf.tile(
            x,
            [1, self.num_points, 1],
            name='global_feats'
        )
        # Concatenate features for point-wise classification
        x = tf.keras.layers.Concatenate(name='full_feats')(
            self.pretransf_feats +
            [self.transf_feats] +
            self.postransf_feats[:-1] +
            [x]
        )
        # Convolve point-wise features
        x = PointNet.build_conv_block(
            x, filters=self.num_pwise_feats, name='pwise_feats'
        )
        return x

    def build_output(self, x, **kwargs):
        """
        Build the output layer of a PointNet neural network for point-wise
        classification tasks.

        See :meth:`architecture.Architecture.build_output`.

        :param x: The input for the output layer.
        :type x: :class:`tf.Tensor`
        :return: The output layer.
        :rtype: :class:`tf.Tensor`
        """
        # Handle output layer for binary crossentropy loss
        if self.binary_crossentropy:
            return tf.keras.layers.Conv1D(
                1,
                kernel_size=1,
                activation='sigmoid',
                name='pwise_out'
            )(x)
        # Handle output layer for the general case
        return tf.keras.layers.Conv1D(
            self.num_classes,
            kernel_size=1,
            activation='softmax',
            name='pwise_out'
        )(x)
