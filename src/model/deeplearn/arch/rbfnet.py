# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.dlrun.point_net_pre_processor import \
    PointNetPreProcessor
from src.model.deeplearn.dlrun.point_net_post_processor import \
    PointNetPostProcessor
from src.model.deeplearn.layer.rbf_feat_extract_layer import \
    RBFFeatExtractLayer
from src.model.deeplearn.layer.features_structuring_layer import \
    FeaturesStructuringLayer
from src.utils.dict_utils import DictUtils
from src.main.main_config import VL3DCFG
import tensorflow as tf


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
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'RBFNet'
        super().__init__(**kwargs)
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['MODEL']['RBFNet']
        )
        # Assign attributes
        self.fnames = kwargs.get('fnames', None)
        # Update the preprocessing logic
        self.pre_runnable = PointNetPreProcessor(**kwargs['pre_processing'])
        # Update the postprocessing logic
        self.post_runnable = PointNetPostProcessor(self.pre_runnable)
        # The number of points (cells for grid, points for furth. pt. sampling)
        self.num_points = self.pre_runnable.get_num_input_points()
        # The specification for the RBF feature extraction layer
        self.rbfs = kwargs['rbfs']
        # The specification for the FSL and RBFFeatProcessing layers
        self.feature_structuring = kwargs.get('feature_structuring', None)
        self.feature_processing = kwargs.get('feature_processing', None)
        # Neural network architecture specifications
        self.tnet_pre_filters_spec = kwargs['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = kwargs['tnet_post_filters_spec']
        self.tnet_kernel_initializer = kwargs.get(
            'tnet_kernel_initializer', 'glorot_normal'
        )
        self.enhanced_dim = kwargs.get('enhanced_dim', [16, 128, 1024])
        self.enhancement_kernel_initializer = kwargs.get(
            'enhancement_kernel_initializer', 'glorot_normal'
        )
        # Initialize cache-like attributes
        self.rbf_layers, self.rbf_output_tensors = [], []
        self.rbf_feat_proc_layer = None
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
            self.X,
            num_features=3,
            name='input_transf',
            tnet_pre_filters=self.tnet_pre_filters_spec,
            tnet_post_filters=self.tnet_post_filters_spec,
            kernel_initializer=self.tnet_kernel_initializer
        )
        self.Xtransf = x
        # RBF feature extraction layers
        for i, rbf in enumerate(self.rbfs):
            # Generate RBF output
            rbf_layer = RBFFeatExtractLayer(
                max_radii=rbf['max_radii'],
                radii_resolution=rbf['radii_resolution'],
                angular_resolutions=rbf['angular_resolutions'],
                kernel_function_type=rbf['kernel_function_type'],
                structure_initialization_type=rbf['structure_initialization_type'],
                trainable_Q=rbf['trainable_kernel_structure'],
                trainable_omega=rbf['trainable_kernel_sizes'],
                name=f'RBF_feat_extract{i+1}'
            )
            _x = rbf_layer(x)
            if rbf.get('batch_normalization', False):
                _x = tf.keras.layers.BatchNormalization(
                    momentum=0.0, name=f'RBF_feat_extract_BN{i+1}'
                )(_x)
            if rbf.get('activation', None) is not None:
                act = rbf['activation']
                act_low = act.lower()
                if act_low == 'relu':
                    _x = tf.keras.layers.Activation(
                        "relu", name=f'RBF_feat_extract_ReLU{i+1}'
                    )(_x)
                else:
                    raise DeepLearningException(
                        f'RBFNet does not support "{act}" activation for RBFs.'
                    )
            # Store the RBF layer itself
            self.rbf_layers.append(rbf_layer)
            # Structure RBF output
            if self.check_feature_structuring('fsl_rbf_features_dim_out'):
                _x = self.build_FSL_block(
                    _x,
                    self.feature_structuring,
                    self.feature_structuring['fsl_rbf_features_dim_out'],
                    f'rbf{i+1}_feats'
                )
            # Store RBF output
            self.rbf_output_tensors.append(_x)
        x = tf.keras.layers.Concatenate(name='rbf_concat')(
            self.rbf_output_tensors
        )
        # Apply enhancement if requested
        if self.enhanced_dim is not None:
            for i, enhanced_dim in enumerate(self.enhanced_dim):
                x = PointNet.build_mlp_block(
                    x,
                    enhanced_dim,
                    f'enhancement{i+1}',
                    self.enhancement_kernel_initializer
                )
            # Structure enhanced RBF output
            if self.check_feature_structuring(
                'fsl_rbf_enhanced_features_dim_out'
            ):
                x = self.build_FSL_block(
                    x,
                    self.feature_structuring,
                    self.feature_structuring['fsl_rbf_enhanced_features_dim_out'],
                    name='rbf_enhanced_feats'
                )
        # Return
        return x

    # ---  RBFNET PWISE CLASSIF METHODS  --- #
    # -------------------------------------- #
    def build_FSL_block(self, F, fs, dim_out, name):
        """
        Assist the building of feature structuring blocks providing the common
        operations.

        See :class:`.FeaturesStructuringLayer`.

        :param F: The tensor of input features.
        :param fs: The feature structuring specification.
        :param dim_out: The output dimensionality for the FSL block.
        :return: The built FSL block
        """
        X = self.Xtransf if fs['transformed_structure'] else self.X
        if isinstance(dim_out, str) and dim_out.lower() == 'dim_in':
            dim_out = F.shape[-1]
        fsl = FeaturesStructuringLayer(
            max_radii=fs['max_radii'],
            radii_resolution=len(fs['angular_resolutions']),
            angular_resolutions=fs['angular_resolutions'],
            structure_dimensionality=3,
            dim_out=dim_out,
            concatenation_strategy=fs['concatenation_strategy'],
            trainable_QX=fs['trainable_QX'],
            trainable_QW=fs['trainable_QW'],
            trainable_omegaD=fs['trainable_omegaD'],
            trainable_omegaF=fs['trainable_omegaF'],
            name=f'fsl_{name}'
        )([X, F])
        if fs.get('enhance', False):
            fsl = PointNet.build_mlp_block(
                fsl,
                dim_out,
                f'fsl_{name}_enhancement',
                self.enhancement_kernel_initializer
            )
        return fsl

    def check_feature_structuring(self, dim_out_key):
        """
        Check whether the feature structuring specification supports the
        given key (True) or not (False).

        :param dim_out_key: The key of the output dimensionaliy element to
            be checked to decide on the feature structuring availability.
        :return: True if the feature structuring is supported for given key,
            false otherwise.
        """
        return (
            self.feature_structuring is not None and
            self.feature_structuring.get(dim_out_key, 0) != 0
        )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized RBFNet architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add RBFNet's attributes to state dictionary
        state['fnames'] = self.fnames
        state['num_points'] = self.num_points
        state['rbfs'] = self.rbfs
        state['feature_structuring'] = self.feature_structuring
        state['feature_processing'] = self.feature_processing
        state['enhanced_dim'] = self.enhanced_dim
        state['enhancement_kernel_initializer'] = \
            self.enhancement_kernel_initializer
        state['tnet_pre_filters_spec'] = self.tnet_pre_filters_spec
        state['tnet_post_filters_spec'] = self.tnet_post_filters_spec
        state['tnet_kernel_initializer'] = self.tnet_kernel_initializer
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized RBFNet architecture.

        :param state: The state's dictionary of the saved RBFNet architecutre.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign RBFNet's attributes from state dictionary
        self.fnames = state.get('fnames', None)
        self.num_points = state['num_points']
        self.rbfs = state['rbfs']
        self.feature_structuring = state['feature_structuring']
        self.feature_processing = state['feature_processing']
        self.enhanced_dim = state['enhanced_dim']
        self.enhancement_kernel_initializer = \
            state['enhancement_kernel_initializer']
        self.tnet_pre_filters_spec = state['tnet_pre_filters_spec']
        self.tnet_post_filters_spec = state['tnet_post_filters_spec']
        self.tnet_kernel_initializer = state['tnet_kernel_initializer']
        # Call parent's set state
        super().__setstate__(state)
        # Track rbf layers
        self.rbf_layers = [
            layer
            for layer in self.nn.layers
            if isinstance(layer, RBFFeatExtractLayer)
        ]
