# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_post_processor import \
    HierarchicalFPSPostProcessor
from src.model.deeplearn.layer.features_downsampling_layer import \
    FeaturesDownsamplingLayer
from src.model.deeplearn.layer.features_upsampling_layer import \
    FeaturesUpsamplingLayer
from src.model.deeplearn.layer.grouping_point_net_layer import \
    GroupingPointNetLayer
from src.utils.dl_utils import DLUtils
import tensorflow as tf


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
        See :meth:`architecture.Architecture.__init__`.
        """
        # Call parent's init
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'ConvAutoenc_PointWise_Classification'
        super().__init__(**kwargs)
        # Assign attributes
        self.fnames = kwargs.get('fnames', None)
        if self.fnames is None:
            self.fnames = ['ones']  # If no features are given, use ones
        self.pre_runnable = HierarchicalFPSPreProcessor(
            **kwargs['pre_processing']
        )
        self.post_runanble = HierarchicalFPSPostProcessor(self.pre_runnable)
        self.num_classes = kwargs.get('num_classes', None)
        self.feature_extraction = kwargs.get('feature_extraction', None)
        self.num_downsampling_neighbors = \
            self.pre_runnable.num_downsampling_neighbors
        self.num_pwise_neighbors = \
            self.pre_runnable.num_pwise_neighbors
        self.num_upsampling_neighbors = \
            self.pre_runnable.num_upsampling_neighbors
        self.downsampling_filter = kwargs.get(
            'downsampling_filter', 'mean'
        )
        self.upsampling_filter = kwargs.get(
            'upsampling_strategy', 'mean'
        )
        self.upsampling_bn = kwargs.get('upsampling_bn', True)
        self.upsampling_bn_momentum = kwargs.get(
            'upsampling_bn_momentum', 0.0
        )
        self.conv1d_kernel_initializer = kwargs.get(
            'kernel_initializer', 'glorot_normal'
        )
        self.output_kernel_initializer = kwargs.get(
            'kernel_initializer', 'glorot_normal'
        )
        self.max_depth = len(self.num_downsampling_neighbors)
        self.binary_crossentropy = False
        comp_args = kwargs.get('compilation_args', None)
        self.binary_crossentropy = DLUtils.is_using_binary_crossentropy(
            comp_args, default=False
        )
        # Cache-like attributes
        self.Xs = None
        self.F = None
        self.NDs, self.Ns, self.NUs = [None]*3
        self.skip_links = None
        self.last_downsampling_tensor = None
        self.last_upsampling_tensor = None

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_input(self):
        r"""
        Build the input layer of the neural network. A convolutional
        autoencoder expects to receive many input tensors representing the
        hierarchical nature of the architecture. More concretely, for each
        element in the batch there must be:

        1)  The structure space matrices representing the points in the
            hierarchy of FPS receptive fields (typically, :math:`n_x=3`,
            i.e., 3D point clouds).

        .. math::
            \pmb{X}_1 \in \mathbb{R}^{R_1 \times n_x}, \ldots,
            \pmb{X}_{d^*} \in \mathbb{R}^{R_{d^*} \times n_x}

        2)  The feature space matrix representing the points in the first
            receptive field of the hierarchy.

        .. math::
            \pmb{F}_1 \in \mathbb{R}^{R_1 \times n_f}

        3)  The downsampling matrices after the first one (which is not used by
            the neural network itself but immediately before to transform the
            original input to the first receptive field).

        .. math::
            \pmb{N}^D_2 \in \mathbb{Z}^{R_2 \times K^D_2}, \ldots,
            \pmb{N}^D_{d^*} \in \mathbb{Z}^{R_{d^*} \times K^D_{d^*}}

        4)  The point-wise neighborhood matrices to be used at each downsampled
            representation as topological information.

        .. math::
            \pmb{N}_2 \in \mathbb{Z}^{R_2 \times K_2}, \ldots,
            \pmb{N}_{d^*} \in \mathbb{Z}^{R_{d^*} \times K_{d^*}}

        3)  The upsampling matrices after the first one (which is not used by
            the neural network itself but immediately after to transform the
            output from the first receptive field to the original space).

        .. math::
            \pmb{N}^U_2 \in \mathbb{Z}^{R_2 \times K^U_2}, \ldots,
            \pmb{N}^U_{d^*} \in \mathbb{Z}^{R_{d^*} \times K^U_{d^*}}

        :return: Built layer.
        :rtype: :class:`tf.Tensor`
        """
        # Handle coordinates as input
        self.Xs = [
            tf.keras.layers.Input(shape=(None, 3), name=f'X_{d+1}')
            for d in range(self.max_depth)
        ]
        # Handle input features
        self.F = tf.keras.layers.Input(
            shape=(None, len(self.fnames)),
            name='Fin'
        )
        # Handle downsampling matrices
        self.NDs = [
            tf.keras.layers.Input(
                shape=(None, self.num_downsampling_neighbors[d]),
                dtype='int32',
                name=f'ND_{d+1}'
            )
            for d in range(1, self.max_depth)
        ]
        # Handle point-wise neighborhood matrices
        self.Ns = [
            tf.keras.layers.Input(
                shape=(None, self.num_pwise_neighbors[d]),
                dtype='int32',
                name=f'N_{d+1}'
            )
            for d in range(self.max_depth)
        ]
        # Handle upsampling matrices
        self.NUs = [
            tf.keras.layers.Input(
                shape=(None, self.num_upsampling_neighbors[d]),
                dtype='int32',
                name=f'NU_{d+1}'
            )
            for d in range(1, self.max_depth)
        ]
        # Return list of inputs
        return [self.Xs, self.F, self.NDs, self.Ns, self.NUs]

    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the convolutional autoencoder neural
        network.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer.
        :rtype: :class.`.tf.Tensor`
        """
        # Downsampling hierarchy
        self.build_downsampling_hierarchy()
        # Upsampling hierarchy
        self.build_upsampling_hierarchy()
        # Return last hidden layer
        return self.last_upsampling_tensor

    def build_output(self, x, **kwargs):
        """
        Build the output layer of the convolutional autoencoder neural network
        for point-wise classification tasks.

        See :meth:`architecture.Architecture.build_output`.
        """
        # Handle output layer for binary cross-entropy loss
        if self.binary_crossentropy:
            return tf.keras.layers.Conv1D(
                1,
                kernel_size=1,
                activation='sigmoid',
                kernel_initializer=self.output_kernel_initializer,
                name='pwise_out'
            )(x)
        # Handle output layer for the general case
        return tf.keras.layers.Conv1D(
            self.num_classes,
            kernel_size=1,
            activation='softmax',
            kernel_initializer=self.output_kernel_initializer,
            name='pwise_out'
        )(x)

    # ---  CONVOLUTIONAL AUTOENCODER PWISE CLASSIF METHODS  --- #
    # --------------------------------------------------------- #
    def build_downsampling_hierarchy(self):
        """
        Build the downsampling hierarchy.

        :return: The last layer of the downsampling hierarchy.
        :rtype: :class:`tf.Tensor`
        """
        feat_extract_type = self.feature_extraction['type']
        feat_extract_type_low = feat_extract_type.lower()
        if feat_extract_type_low == 'pointnet':
            self.build_downsampling_pnet_hierarchy()
        elif feat_extract_type_low == 'kpconv':
            self.build_downsampling_kpconv_hierarchy()
        else:
            raise DeepLearningException(
                f'ConvAutoencPwiseClassif received a "{feat_extract_type}" '
                'as type of feature extraction. It is not supported.'
            )

    def build_downsampling_pnet_hierarchy(self):
        """
        Build the downsampling hierarchy based on the PointNet operator.
        """
        self.skip_links = []
        i = 0
        ops_per_depth = self.feature_extraction['operations_per_depth']
        x = self.F
        for _ in range(ops_per_depth[0]):
            x = GroupingPointNetLayer(
                self.feature_extraction['feature_space_dims'][i],
                H_activation=self.feature_extraction['H_activation'][i],
                H_initializer=self.feature_extraction['H_initializer'][i],
                H_regularizer=self.feature_extraction['H_regularizer'][i],
                H_constraint=self.feature_extraction['H_constraint'][i],
                gamma_activation=self.feature_extraction['gamma_activation'][i],
                gamma_kernel_initializer=self.feature_extraction[
                    'gamma_kernel_initializer'
                ][i],
                gamma_kernel_regularizer=self.feature_extraction[
                    'gamma_kernel_regularizer'
                ][i],
                gamma_kernel_constraint=self.feature_extraction[
                    'gamma_kernel_constraint'
                ][i],
                gamma_bias_enabled=self.feature_extraction[
                    'gamma_bias_enabled'
                ][i],
                gamma_bias_initializer=self.feature_extraction[
                    'gamma_bias_initializer'
                ][i],
                gamma_bias_regularizer=self.feature_extraction[
                    'gamma_bias_regularizer'
                ][i],
                gamma_bias_constraint=self.feature_extraction[
                    'gamma_bias_constraint'
                ][i],
                name=f'GPNet_d1_{i+1}'
            )([self.Xs[0], x, self.Ns[0]])
            i += 1
        self.skip_links.append(x)
        for d in range(self.max_depth-1):
            x = FeaturesDownsamplingLayer(
                filter=self.downsampling_filter,
                name=f'DOWN_d{d+2}'
            )([
                self.Xs[d], self.Xs[d+1], x, self.NDs[d]
            ])
            for _ in range(ops_per_depth[d+1]):
                x = GroupingPointNetLayer(
                    self.feature_extraction['feature_space_dims'][i],
                    H_activation=self.feature_extraction['H_activation'][i],
                    H_initializer=self.feature_extraction['H_initializer'][i],
                    H_regularizer=self.feature_extraction['H_regularizer'][i],
                    H_constraint=self.feature_extraction['H_constraint'][i],
                    gamma_activation=self.feature_extraction['gamma_activation'][i],
                    gamma_kernel_initializer=self.feature_extraction[
                        'gamma_kernel_initializer'
                    ][i],
                    gamma_kernel_regularizer=self.feature_extraction[
                        'gamma_kernel_regularizer'
                    ][i],
                    gamma_kernel_constraint=self.feature_extraction[
                        'gamma_kernel_constraint'
                    ][i],
                    gamma_bias_enabled=self.feature_extraction[
                        'gamma_bias_enabled'
                    ][i],
                    gamma_bias_initializer=self.feature_extraction[
                        'gamma_bias_initializer'
                    ][i],
                    gamma_bias_regularizer=self.feature_extraction[
                        'gamma_bias_regularizer'
                    ][i],
                    gamma_bias_constraint=self.feature_extraction[
                        'gamma_bias_constraint'
                    ][i],
                    name=f'GPNet_d{d+2}_{i+1}'
                )([self.Xs[d+1], x, self.Ns[d+1]])
                i += 1
            self.skip_links.append(x)
        self.last_downsampling_tensor = x

    def build_downsampling_kpconv_hierarchy(self):
        """
        Build the downsampling hierarchy based on the KPConv operator.
        """
        raise DeepLearningException(
            'ConvAutoencPwiseClassif does not support KPConv-based '
            'downsampling hierarchies YET.'
        )

    def build_upsampling_hierarchy(self):
        """
        Build the upsampling hierarchy.

        :return: The last layer of the upsampling hierarchy.
        :rtype: :class:`tf.Tensor`
        """
        x = self.last_downsampling_tensor
        for d in range(self.max_depth-1):
            reverse_d = self.max_depth-2-d
            skip_link = self.skip_links[reverse_d]
            # Upsampling layer itself
            x = FeaturesUpsamplingLayer(
                filter=self.upsampling_filter,
                name=f'UP_d{reverse_d+2}'
            )([self.Xs[reverse_d], self.Xs[reverse_d-1], x, self.NUs[reverse_d]])
            x = tf.keras.layers.Concatenate(
                name=f'CONCAT_d{reverse_d+1}'
            )([x, skip_link])
            # 1D convolutions after upsampling
            filters = self.feature_extraction['feature_space_dims'][reverse_d]
            x = tf.keras.layers.Conv1D(
                filters,
                kernel_size=1,
                strides=1,
                padding="valid",
                kernel_initializer=self.conv1d_kernel_initializer,
                name=f'Conv1D_d{reverse_d+1}'
            )(x)
            if self.upsampling_bn:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.upsampling_bn_momentum,
                    name=f'BN_d{reverse_d+1}'
                )(x)
            x = tf.keras.layers.Activation("relu")(x)
        self.last_upsampling_tensor = x
