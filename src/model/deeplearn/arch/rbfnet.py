# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.model.deeplearn.regularizer.features_orthogonal_regularizer import \
    FeaturesOrthogonalRegularizer
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.dlrun.point_net_pre_processor import \
    PointNetPreProcessor
from src.model.deeplearn.dlrun.point_net_post_processor import \
    PointNetPostProcessor
from src.utils.dict_utils import DictUtils
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class RBFNet(Architecture, ABC):
    """
    :author: Alberto M. Esmoris Pena
    
    The Radial Basis Function Net (RBFNet) architecture.
    
    See https://arxiv.org/abs/1812.04302
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:architecture.Architecture.__init__`.
        """
        # Call parent's init
        kwargs['arch_name'] = 'RBFNet'
        super().__init__(**kwargs)
        # Update the preprocessing logic
        self.pre_runnable = PointNetPreProcessor(**kwargs['pre_processing'])
        # Update the postprocessing logic
        self.post_runnable = PointNetPostProcessor(self.pre_runnable)
        # The number of points (cells for grid, points for furth. pt. sampling)
        self.num_points = self.pre_runnable.get_num_input_points()
        # Neural network architecture specifications
        self.tnet_pre_filters_spec = kwargs['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = kwargs['tnet_post_filters_spec']
        self.tnet_kernel_initializer = kwargs.get(
            'tnet_kernel_initializer', 'glorot_normal'
        )
        self.enhanced_dim = kwargs.get('enhanced_dim', 1024)
        self.enhancement_kernel_initializer = kwargs.get(
            'enhancement_kernel_initializer', 'glorot_normal'
        )
        self.after_features_MLPs = kwargs.get(
            'after_features_MLPs', [512, 128, 1]
        )
        self.after_features_kernel_initializer = kwargs.get(
            'after_features_kernel_initializer', 'glorot_normal'
        )

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
        Build the hidden layers of the RBFNet neural network.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer.
        :rtype: :class:`.tf.Tensor`
        """
        # Input transformation block
        x = PointNet.build_transformation_block(
            x,
            num_features=3,
            name='input_transf',
            tnet_pre_filters=self.tnet_pre_filters_spec,
            tnet_post_filters=self.tnet_post_filters_spec,
            kernel_initializer=self.tnet_kernel_initializer
        )
        # RBF feature extraction layer
        # TODO Rethink : Implement
        # Apply enhancement if requested
        if self.enhanced_dim > 0:
            x = PointNet.build_mlp_block(
                x,
                self.enhanced_dim,
                'enhancement',
                self.enhancement_kernel_initializer
            )
        # Pooling
        x = tf.keras.layers.GlobalMaxPooling1D(
            name=f'after_feats_pooling'
        )(x)
        # MLPs
        for i, dim in self.after_features_MLPs:
            x = PointNet.build_mlp_block(
                x,
                dim,
                f'after_feats_MLP{i+1}',
                self.after_features_kernel_initializer
            )
        # Return
        return x