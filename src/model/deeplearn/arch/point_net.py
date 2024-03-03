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
from src.main.main_config import VL3DCFG
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
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['MODEL']['PointNet']
        )
        # Set feature names
        self.fnames = kwargs.get('fnames', None)
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
        self.pretransf_feats_F_spec = kwargs.get(
            'pretransf_feats_F_spec', None
        )
        self.postransf_feats_F_spec = kwargs.get('postransf_feats_F_spec', None)
        self.tnet_pre_filters_F_spec = kwargs.get('tnet_pre_filters_F_spec', None)
        self.tnet_post_filters_F_spec = kwargs.get(
            'tnet_post_filters_F_spec', None
        )
        self.kernel_initializer_F = kwargs.get(
            'kernel_initializer_F', 'glorot_normal'
        )
        self.skip_link_features_X = kwargs.get('skip_link_features', False)
        self.include_pretransf_feats_X = kwargs.get(
            'include_pretransf_feats_X', False
        )
        self.include_transf_feats_X = kwargs.get(
            'include_transf_feats_X', True
        )
        self.include_postransf_feats_X = kwargs.get(
            'include_postransf_feats_X', False
        )
        self.include_global_feats_X = kwargs.get(
            'include_global_feats_X', True
        )
        self.skip_link_features_F = kwargs.get(
            'skip_link_features_F', False
        )
        self.include_pretransf_feats_F = kwargs.get(
            'include_pretransf_feats_F', False
        )
        self.include_transf_feats_F = kwargs.get(
            'include_transf_feats_F', True
        )
        self.include_postransf_feats_F = kwargs.get(
            'include_postransf_feats_F', False
        )
        self.include_global_feats_F = kwargs.get(
            'include_global_feats_F', True
        )
        # Initialize cache-like attributes
        self.pretransf_feats_X, self.pretransf_feats_F = None, None
        self.postransf_feats_X, self.postransf_feats_F = None, None
        self.transf_feats_X, self.transf_feats_F = None, None
        self.X, self.F = None, None
        self.Xtransf, self.Ftransf = None, None

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
        return PointNet.build_point_net_input(self)

    @staticmethod
    def build_point_net_input(pnet):
        """
        See :meth:`PointNet.build_input`.
        """
        pnet.X = tf.keras.layers.Input(shape=(None, 3), name='Xin')
        # Handle input features, if any
        if pnet.fnames is not None:
            pnet.F = tf.keras.layers.Input(
                shape=(None, len(pnet.fnames)),
                name='Fin'
            )
        # Return
        if pnet.F is None:
            return pnet.X
        return [pnet.X, pnet.F]

    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the PointNet neural network.

        :param x: The input layer for the first hidden layer. Alternatively,
            it can be a list with many input layers.
        :type x: :class:`tf.Tensor` or list
        :return: The last hidden layer. Alternatively, it can be a list with
            many hidden layers.
        :rtype: :class:`tf.Tensor` or list
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
        # Build the PointNet block on the structure space
        hidden_X, pretransf_feats_X, transf_feats_X, postransf_feats_X, \
            self.Xtransf = PointNet.build_hidden_pointnet(self.X, **_kwargs)
        # Update cached feature layers for the structure space
        self.pretransf_feats_X = pretransf_feats_X
        self.transf_feats_X = transf_feats_X
        self.postransf_feats_X = postransf_feats_X
        # Handle feature space when requested
        if self.F is not None:
            # Update the input dictionary if necessary
            pretransf_feats_F = kwargs.get(
                'pretransf_feats_F', self.pretransf_feats_F_spec
            )
            if pretransf_feats_F is not None:
                _kwargs['pretransf_feats'] = pretransf_feats_F
            postransf_feats_F = kwargs.get(
                'postransf_feats_F', self.postransf_feats_F_spec
            )
            if postransf_feats_F is not None:
                _kwargs['postransf_feats'] = postransf_feats_F
            tnet_pre_filters_F = kwargs.get(
                'tnet_pre_filters_F', self.tnet_pre_filters_F_spec
            )
            if tnet_pre_filters_F is not None:
                _kwargs['tnet_pre_filters'] = tnet_pre_filters_F
            tnet_post_filters_F = kwargs.get(
                'tnet_post_filters_F', self.tnet_post_filters_F_spec
            )
            if tnet_post_filters_F is not None:
                _kwargs['tnet_post_filters'] = tnet_post_filters_F
            kernel_initializer_F = kwargs.get(
                'kernel_initializer_F', self.kernel_initializer_F
            )
            if kernel_initializer_F is not None:
                _kwargs['kernel_initializer'] = kernel_initializer_F
            # Build the PointNet block on the feature space
            hidden_F, pretransf_feats_F, transf_feats_F, postransf_feats_F, \
                self.Ftransf = PointNet.build_hidden_pointnet(
                    self.F,
                    **_kwargs,
                    space_dimensionality=len(self.fnames),
                    prefix='F_'
                )
            # Update cached feature layers for the feature space
            self.pretransf_feats_F = pretransf_feats_F
            self.transf_feats_F = transf_feats_F
            self.postransf_feats_F = postransf_feats_F
            # Return list of hidden layers (one structure, one features)
            return [hidden_X, hidden_F]
        # Return last hidden layer
        return hidden_X

    # ---  POINTNET METHODS  --- #
    # -------------------------- #
    @staticmethod
    def build_hidden_pointnet(
        x,
        pretransf_feats,
        postransf_feats,
        tnet_pre_filters,
        tnet_post_filters,
        kernel_initializer,
        space_dimensionality=3,
        prefix='X_'
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
        :param space_dimensionality: The dimensionality of the space where
            the PointNet block operates (typicalle 3D for the structure
            space, i.e., x,y,z).
        :type space_dimensionality: int
        :return: Last layer of the built PointNet, the list of
            pre-transformations, the layer of transformed features, and the
            list of post-transformations. Finally, the aligned input tensor.
        :rtype: :class:`tf.Tensor` and list and :class:`tf.keras.Layer` and list
            and :class:`tf.Tensor`
        """
        # First transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=space_dimensionality,
            name=f'{prefix}input_transf',
            tnet_pre_filters=tnet_pre_filters,
            tnet_post_filters=tnet_post_filters,
            kernel_initializer=kernel_initializer
        )
        aligned_x = x
        # Features before the second transformation block
        pretransf_feat_layers = []
        for pretransf_feat_spec in pretransf_feats:
            x = PointNet.build_conv_block(
                x,
                filters=pretransf_feat_spec['filters'],
                kernel_initializer=kernel_initializer,
                name=f'{prefix}{pretransf_feat_spec["name"]}'
            )
            pretransf_feat_layers.append(x)
        # The second transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=pretransf_feats[-1]['filters'],
            name=f'{prefix}hidden_transf',
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
                name=f'{prefix}{postransf_feat_spec["name"]}'
            )
            postransf_feat_layers.append(x)
        return x, pretransf_feat_layers, transf_feats, postransf_feat_layers,\
            aligned_x

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
        state['fnames'] = self.fnames
        state['num_points'] = self.num_points
        state['kernel_initializer'] = self.kernel_initializer
        state['pretransf_feats_spec'] = self.pretransf_feats_spec
        state['postransf_feats_spec'] = self.postransf_feats_spec
        state['tnet_pre_filters_spec'] = self.tnet_pre_filters_spec
        state['tnet_post_filters_spec'] = self.tnet_post_filters_spec
        state['kernel_initializer_F'] = self.kernel_initializer_F
        state['pretransf_feats_F_spec'] = self.pretransf_feats_F_spec
        state['postransf_feats_F_spec'] = self.postransf_feats_F_spec
        state['tnet_pre_filters_F_spec'] = self.tnet_pre_filters_F_spec
        state['tnet_post_filters_F_spec'] = self.tnet_post_filters_F_spec
        state['skip_link_features_X'] = self.skip_link_features_X
        state['include_pretransf_feats_X'] = self.include_pretransf_feats_X
        state['include_transf_feats_X'] = self.include_transf_feats_X
        state['include_postransf_feats_X'] = self.include_postransf_feats_X
        state['include_global_feats_X'] = self.include_global_feats_X
        state['skip_link_features_F'] = self.skip_link_features_F
        state['include_pretransf_feats_F'] = self.include_pretransf_feats_F
        state['include_transf_feats_F'] = self.include_transf_feats_F
        state['include_postransf_feats_F'] = self.include_postransf_feats_F
        state['include_global_feats_F'] = self.include_global_feats_F
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
        self.fnames = state.get('fnames', None)
        self.num_points = state['num_points']
        self.kernel_initializer = state['kernel_initializer']
        self.pretransf_feats_spec = state['pretransf_feats_spec']
        self.postransf_feats_spec = state['postransf_feats_spec']
        self.tnet_pre_filters_spec = state['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = state['tnet_post_filters_spec']
        self.kernel_initializer_F = state['kernel_initializer_F']
        self.pretransf_feats_F_spec = state['pretransf_feats_F_spec']
        self.postransf_feats_F_spec = state['postransf_feats_F_spec']
        self.tnet_pre_filters_F_spec = state['tnet_pre_filters_F_spec']
        self.tnet_post_filters_F_spec = state['tnet_post_filters_F_spec']
        self.skip_link_features_X = state['skip_link_features_X']
        self.include_pretransf_feats_X = state['include_pretransf_feats_X']
        self.include_transf_feats_X = state['include_transf_feats_X']
        self.include_postransf_feats_X = state['include_postransf_feats_X']
        self.include_global_feats_X = state['include_global_feats_X']
        self.skip_link_features_F = state['skip_link_features_F']
        self.include_pretransf_feats_F = state['include_pretransf_feats_F']
        self.include_transf_feats_F = state['include_transf_feats_F']
        self.include_postransf_feats_F = state['include_postransf_feats_F']
        self.include_global_feats_F = state['include_global_feats_F']
        self.features_structuring_layer = state.get(
            'features_structuring_layer', None
        )
        # Call parent's set state
        super().__setstate__(state)
