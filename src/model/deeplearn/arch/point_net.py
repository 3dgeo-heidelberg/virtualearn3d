# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.model.deeplearn.layer.orthogonal_regularizer import \
    OrthogonalRegularizer
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.point_net_pre_processor import \
    PointNetPreProcessor
from src.utils.dict_utils import DictUtils
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PointNet(Architecture, ABC):
    """
    :author: Alberto M. Esmoris Pena

    The PointNet architecture.

    See https://arxiv.org/abs/1612.00593 and
    https://keras.io/examples/vision/pointnet_segmentation/#pointnet-model
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:`architecture.Architecture.__init__`.
        """
        # Call parent's init
        kwargs['arch_name'] = 'PointNet'
        super().__init__(**kwargs)
        # Assign the attributes of the PointNet architecture
        self.num_points = kwargs.get('num_points', None)
        if self.num_points is None:
            raise DeepLearningException(
                'The PointNet architecture instantiation requires '
                'the number of points because it works with a fixed input '
                'size. None was given.'
            )
        # Initialize cache-like attributes
        self.pretransf_feats, self.postransf_feats = [None]*2
        self.transf_feats = None
        # Update the preprocessing logic
        self.pre_runnable = PointNetPreProcessor()  # TODO Rethink : kwargs
        # Update the postprocessing logic
        #self.post_runnable(PointNetPostProcessor(self.pre_runnable))  # TODO Rethink : Implement

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_input(self):
        """
        Build the input layer of the neural network. By default, only the
        3D coordinates are considered as input, i.e., input dimensionality is
        three.

        See :meth:`architecture.Architecture.build_input`.

        :return: Built layer.
        :rtype: :class:`tf.Tensor`
        """
        return tf.keras.layers.Input(shape=(None, 3))

    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the PointNet neural network.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer.
        :rtype: :class:`tf.Tensor`
        """
        # Build the input dictionary for the build_hidden_pointnet function
        _kwargs = {
            'pretransf_feats': kwargs.get('pretransf_feats', None),
            'postransf_feats': kwargs.get('postransf_feats', None)
        }
        # Remove None from dictionary to prevent overriding defaults
        _kwargs = DictUtils.delete_by_val(_kwargs, None)
        # Build the PointNet block
        hidden, pretransf_feats, transf_feats, postransf_feats = \
            PointNet.build_hidden_pointnet(x, **_kwargs)
        # Update cached feature layers
        self.pretransf_feats = pretransf_feats
        self.transf_feats = transf_feats
        self.postransf_feats = postransf_feats
        # Return last hidden layer
        return hidden

    # ---  POINTNET METHODS  --- #
    # -------------------------- #
    @staticmethod
    def build_hidden_pointnet(
        x,
        pretransf_feats=(
            {
                'filters': 64,
                'name': 'feats_64'
            },
            {
                'filters': 128,
                'name': 'feats_128_A'
            },
            {
                'filters': 128,
                'name': 'feats_128_B'
            }
        ),
        postransf_feats=(
            {
                'filters': 512,
                'name': 'feats_512'
            },
            {
                'filters': 2048,
                'name': 'feats_2018'
            }
        )
    ):
        """
        Build the PointNet block of the architecture on the given input.

        :param x: The input for the PointNet block/architecture.
        :type x: :class:`tf.Tensor`
        :param pretransf_feats: The specification of the filters and the name
            for each feature extraction layer before the transformation block
            in the middle.
        :type pretransf_feats: list
        :param postransf_feats: The specification of the filters and the name
            for each feature extraction layer after the transformation block
            in the middle.
        :type postransf_feats: list
        :return: Last layer of the built PointNet, the list of
            pre-transformations, the layer of transformed features, and the
            list of post-transformations.
        :rtype: :class:`tf.Tensor` and list and :class:`tf.keras.Layer` and list
        """
        # First transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=3,
            name='input_transf'
        )
        # Features before the second transformation block
        pretransf_feat_layers = []
        for pretransf_feat_spec in pretransf_feats:
            x = PointNet.build_conv_block(
                x,
                filters=pretransf_feat_spec['filters'],
                name=pretransf_feat_spec['name']
            )
            pretransf_feat_layers.append(x)
        # The second transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=pretransf_feats[-1]['filters'],
            name='hidden_transf'
        )
        transf_feats = x
        # Features after the second transformation block
        postransf_feat_layers = []
        for postransf_feat_spec in postransf_feats:
            x = PointNet.build_conv_block(
                x,
                filters=postransf_feat_spec['filters'],
                name=postransf_feat_spec['name']
            )
            postransf_feat_layers.append(x)
        return x, pretransf_feat_layers, transf_feats, postransf_feat_layers

    @staticmethod
    def build_transformation_block(inputs, num_features, name):
        """
        Build a transformation block.

        :param inputs: The input tensor.
        :type inputs: :class:`tf.Tensor`
        :param num_features: The number of features to be transformed.
        :type num_features: int
        :param name: The name of the block.
        :type name: str
        :return: The last layer of the transformation block
        """
        transf = PointNet.build_transformation_net(
            inputs, num_features, name=name
        )
        transf = tf.keras.layers.Reshape((num_features, num_features))(transf)
        return tf.keras.layers.Dot(axes=(2, 1), name=f'{name}_mm')([
            inputs, transf
        ])

    @staticmethod
    def build_transformation_net(inputs, num_features, name):
        """
        Assists the :func:`point_net.PointNet.build_transformation_block`
        method.
        """
        x = PointNet.build_conv_block(inputs, filters=64, name=f'{name}_1')
        x = PointNet.build_conv_block(x, filters=128, name=f'{name}_2')
        x = PointNet.build_conv_block(x, filters=1024, name=f'{name}_3')
        x = tf.keras.layers.GlobalMaxPooling1D(name=f'{name}_GMaxPool')(x)
        x = PointNet.build_mlp_block(x, filters=512, name=f'{name}_1_1')
        x = PointNet.build_mlp_block(x, filters=256, name=f'{name}_2_1')
        return tf.keras.layers.Dense(
            num_features*num_features,
            kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant(
                np.eye(num_features).flatten()
            ),
            activity_regularizer=OrthogonalRegularizer(
                num_features=num_features
            ),
            name=f'{name}_final'
        )(x)

    @staticmethod
    def build_conv_block(x, filters, name):
        """
        Build a convolutional block.

        :param x: The input tensor.
        :type x: :class:`tf.Tensor`
        :param filters: The dimensionality of the output.
        :type filters: int
        :param name: The name of the block
        :type name: str
        """
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size=1,
            padding="valid",
            name=f'{name}_conv1D'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.0, name=f'{name}_bn'
        )(x)
        return tf.keras.layers.Activation("relu", name=f'{name}_relu')(x)

    @staticmethod
    def build_mlp_block(x, filters, name):
        """
        Build an MLP block.

        :param x: The input tensor.
        :type x: :class:`tf.Tensor`
        :param filters: The dimensionality of the output.
        :type filters: int
        :param name: The name of the block.
        :type name: str
        """
        x = tf.keras.layers.Dense(filters, name=f'{name}_dense')(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.0, name=f'{name}_bn'
        )(x)
        return tf.keras.layers.Activation("relu", name=f'{name}_relu')(x)
