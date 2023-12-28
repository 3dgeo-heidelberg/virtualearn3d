# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.model.deeplearn.regularizer.features_orthogonal_regularizer import \
    FeaturesOrthogonalRegularizer
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.dlrun.point_net_pre_processor import \
    PointNetPreProcessor
from src.model.deeplearn.dlrun.point_net_post_processor import  \
    PointNetPostProcessor
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
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'PointNet'
        super().__init__(**kwargs)
        # Update the preprocessing logic
        self.pre_runnable = PointNetPreProcessor(**kwargs['pre_processing'])
        # Update the postprocessing logic
        self.post_runnable = PointNetPostProcessor(self.pre_runnable)
        # The number of points (cells for grid, points for furth. pt. sampling)
        self.num_points = self.pre_runnable.get_num_input_points()
        # Neural network architecture specifications
        self.kernel_initializer = kwargs.get(
            "kernel_initializer", "glorot_normal"
        )
        self.pretransf_feats_spec = kwargs['pretransf_feats_spec']
        self.postransf_feats_spec = kwargs['postransf_feats_spec']
        self.tnet_pre_filters_spec = kwargs['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = kwargs['tnet_post_filters_spec']
        self.features_structuring_layer = kwargs.get(
            'features_structuring_layer', None
        )
        # Initialize cache-like attributes
        self.pretransf_feats, self.postransf_feats = [None]*2
        self.transf_feats = None

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
            'pretransf_feats': kwargs.get(
                'pretransf_feats', self.pretransf_feats_spec
            ),
            'postransf_feats': kwargs.get(
                'postransf_feats', self.postransf_feats_spec
            ),
            'tnet_pre_filters': kwargs.get(
                'tnet_pre_filters', self.tnet_pre_filters_spec
            ),
            'tnet_post_filters': kwargs.get(
                'tnet_post_filters', self.tnet_post_filters_spec
            ),
            'kernel_initializer': kwargs.get(
                'kernel_initializer', self.kernel_initializer
            )
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
        pretransf_feats,
        postransf_feats,
        tnet_pre_filters,
        tnet_post_filters,
        kernel_initializer
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
        :param tnet_pre_filters: The list of number of filters (integer)
            defining each convolutional block before the global pooling.
        :type tnet_pre_filters: list
        :param tnet_post_filters: The list of number of filters (integer)
            defining each MLP block after the global pooling.
        :type tnet_post_filters: list
        :param kernel_initializer: The name of the kernel initializer
        :type kernel_initializer: str
        :return: Last layer of the built PointNet, the list of
            pre-transformations, the layer of transformed features, and the
            list of post-transformations.
        :rtype: :class:`tf.Tensor` and list and :class:`tf.keras.Layer` and list
        """
        # First transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=3,
            name='input_transf',
            tnet_pre_filters=tnet_pre_filters,
            tnet_post_filters=tnet_post_filters,
            kernel_initializer=kernel_initializer
        )
        # Features before the second transformation block
        pretransf_feat_layers = []
        for pretransf_feat_spec in pretransf_feats:
            x = PointNet.build_conv_block(
                x,
                filters=pretransf_feat_spec['filters'],
                kernel_initializer=kernel_initializer,
                name=pretransf_feat_spec['name']
            )
            pretransf_feat_layers.append(x)
        # The second transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=pretransf_feats[-1]['filters'],
            name='hidden_transf',
            tnet_pre_filters=tnet_pre_filters,
            tnet_post_filters=tnet_post_filters,
            kernel_initializer=kernel_initializer
        )
        transf_feats = x
        # Features after the second transformation block
        postransf_feat_layers = []
        for postransf_feat_spec in postransf_feats:
            x = PointNet.build_conv_block(
                x,
                filters=postransf_feat_spec['filters'],
                kernel_initializer=kernel_initializer,
                name=postransf_feat_spec['name']
            )
            postransf_feat_layers.append(x)
        return x, pretransf_feat_layers, transf_feats, postransf_feat_layers

    @staticmethod
    def build_transformation_block(
        inputs,
        num_features,
        name,
        tnet_pre_filters,
        tnet_post_filters,
        kernel_initializer
    ):
        """
        Build a transformation block.

        :param inputs: The input tensor.
        :type inputs: :class:`tf.Tensor`
        :param num_features: The number of features to be transformed.
        :type num_features: int
        :param name: The name of the block.
        :type name: str
        :param tnet_pre_filters: The list of number of filters (integer)
            defining each convolutional block before the global pooling.
        :type tnet_pre_filters: list
        :param tnet_post_filters: The list of nubmer of filters (integer)
            defining each MLP block after the global pooling.
        :type tnet_post_filters: list
        :param kernel_initializer: The name of the kernel initializer
        :type kernel_initializer: str
        :return: The last layer of the transformation block
        """
        transf = PointNet.build_transformation_net(
            inputs,
            num_features,
            name=name,
            tnet_pre_filters=tnet_pre_filters,
            tnet_post_filters=tnet_post_filters,
            kernel_initializer=kernel_initializer
        )
        transf = tf.keras.layers.Reshape((num_features, num_features))(transf)
        return tf.keras.layers.Dot(axes=(2, 1), name=f'{name}_mm')([
            inputs, transf
        ])

    @staticmethod
    def build_transformation_net(
        inputs,
        num_features,
        name,
        tnet_pre_filters,
        tnet_post_filters,
        kernel_initializer
    ):
        """
        Assists the :func:`point_net.PointNet.build_transformation_block`
        method.
        """
        x = inputs
        # Compute the filters before the global pooling (pre filters)
        for i, filters in enumerate(tnet_pre_filters):
            x = PointNet.build_conv_block(
                x,
                filters=filters,
                kernel_initializer=kernel_initializer,
                name=f'{name}_pre{i+1}_f{filters}'
            )
        # Global pooling
        x = tf.keras.layers.GlobalMaxPooling1D(name=f'{name}_GMaxPool')(x)
        # Compute the filters after the global pooling (post filters)
        for i, filters in enumerate(tnet_post_filters):
            x = PointNet.build_mlp_block(
                x,
                filters=filters,
                kernel_initializer=kernel_initializer,
                name=f'{name}_post{i+1}_f{filters}'
            )
        return tf.keras.layers.Dense(
            num_features*num_features,
            kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant(
                np.eye(num_features).flatten()
            ),
            activity_regularizer=FeaturesOrthogonalRegularizer(
                num_features=num_features
            ),
            name=f'{name}_final'
        )(x)

    @staticmethod
    def build_conv_block(x, filters, name, kernel_initializer):
        """
        Build a convolutional block.

        :param x: The input tensor.
        :type x: :class:`tf.Tensor`
        :param filters: The dimensionality of the output.
        :type filters: int
        :param name: The name of the block
        :type name: str
        :param kernel_initializer: The name of the kernel initializer
        :type kernel_initializer: str
        """
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size=1,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name=f'{name}_conv1D'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.0, name=f'{name}_bn'
        )(x)
        return tf.keras.layers.Activation("relu", name=f'{name}_relu')(x)

    @staticmethod
    def build_mlp_block(
        x, filters, name, kernel_initializer, batch_normalization=True
    ):
        """
        Build an MLP block.

        :param x: The input tensor.
        :type x: :class:`tf.Tensor`
        :param filters: The dimensionality of the output.
        :type filters: int
        :param kernel_initializer: The name of the kernel initializer
        :type kernel_initializer: str
        :param name: The name of the block.
        :type name: str
        :param batch_normalization: Whether to apply batch normalization (True)
            or not (False).
        :type batch_normalization: bool
        """
        x = tf.keras.layers.Dense(
            filters,
            kernel_initializer=kernel_initializer,
            name=f'{name}_dense'
        )(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization(
                momentum=0.0, name=f'{name}_bn'
            )(x)
        return tf.keras.layers.Activation("relu", name=f'{name}_relu')(x)

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized PointNet architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add PointNet's attributes to state dictionary
        state['num_points'] = self.num_points
        state['kernel_initializer'] = self.kernel_initializer
        state['pretransf_feats_spec'] = self.pretransf_feats_spec
        state['postransf_feats_spec'] = self.postransf_feats_spec
        state['tnet_pre_filters_spec'] = self.tnet_pre_filters_spec
        state['tnet_post_filters_spec'] = self.tnet_post_filters_spec
        state['features_structuring_layer'] = self.features_structuring_layer
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized PointNet architecture.

        :param state: The state's dictionary of the saved PointNet
            architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign PointNet's attributes from state dictionary
        self.num_points = state['num_points']
        self.kernel_initializer = state['kernel_initializer']
        self.pretransf_feats_spec = state['pretransf_feats_spec']
        self.postransf_feats_spec = state['postransf_feats_spec']
        self.tnet_pre_filters_spec = state['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = state['tnet_post_filters_spec']
        self.features_structuring_layer = state.get(
            'features_structuring_layer', None
        )
        # Call parent's set state
        super().__setstate__(state)
