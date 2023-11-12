# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.arch.rbfnet import RBFNet
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.point_net import PointNet
import tensorflow as tf
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class RBFNetPwiseClassif(RBFNet):
    """
    :author: Alberto M. Esmoris Pena

    A specialization of the RBFNet architecture for point-wise classification.

    See :class:`.RBFNet`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:`architecture.RBFNet.__init__`.
        """
        # Call parent's init
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'RBFNet_PointWise_Classification'
        super().__init__(**kwargs)
        # Assign attributes
        self.num_classes = kwargs.get('num_classes', None)
        self.after_features_type = kwargs.get('after_features_type', 'MLP')
        self.after_features_dim = kwargs.get('after_features_dim', [512, 128])
        self.after_features_kernel_initializer = kwargs.get(
            'after_features_kernel_initializer', 'glorot_normal'
        )
        # Neural network architecture specifications
        self.output_kernel_initializer = kwargs.get(
            'output_kernel_initializer', 'glorot_normal'
        )
        # Update the preprocessing logic
        if self.num_classes is None:
            raise DeepLearningException(
                'The RBFNetPwiseClassif architecture instantiation requires '
                'the number of classes defining the problem. None was given.'
            )
        self.binary_crossentropy = False
        comp_args = kwargs.get('compilation_args', None)
        if comp_args is not None:
            loss_args = comp_args.get('loss', None)
            if loss_args is not None:
                fun_name = loss_args.get('function', '').lower()
                self.binary_crossentropy = \
                    fun_name == 'binary_crossentropy' or \
                    fun_name == 'class_weighted_binary_crossentropy'
        # Initialize cache-like attributes
        self.prepool_feats_tensor = None
        self.global_feats_tensor = None

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the RBFNet neural network for point-wise
        classification tasks.

        See :meth:`rbf_net.RBFNet.build_hidden`.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer
        :rtype: :class:`tf.Tensor`
        """
        # Call parent's build hidden
        x = super().build_hidden(x, **kwargs)
        # Compute global features
        self.prepool_feats_tensor = x
        x = tf.keras.layers.MaxPooling1D(
            self.num_points,
            name=f'after_feats_pooling'
        )(x)
        self.global_feats_tensor = tf.tile(
            x,
            [1, self.num_points, 1],
            name='global_feats'
        )
        # Concatenate features for point-wise classification
        x = tf.keras.layers.Concatenate(name='full_feats')(
            self.rbf_output_tensors + [
                self.prepool_feats_tensor,
                self.global_feats_tensor
            ]
        )
        if self.after_features_type.lower() == 'mlp':
            # After features MLPs
            for i, dim in enumerate(self.after_features_dim):
                x = PointNet.build_mlp_block(
                    x,
                    dim,
                    f'after_feats_MLP{i+1}',
                    self.after_features_kernel_initializer
                )
        elif self.after_features_type.lower() == 'conv1d':
            for i, dim in enumerate(self.after_features_dim):
                x = PointNet.build_conv_block(
                    x,
                    filters=dim,
                    name=f'after_feats_Conv1D{i+1}',
                    kernel_initializer=self.after_features_kernel_initializer
                )
        else:
            raise DeepLearningException(
                'RBFNetPwiseClassif received an unexpected after features '
                f'operation type: "{self.after_features_type}"'
            )
        # Return
        return x

    def build_output(self, x, **kwargs):
        """
        Build the output layer of a RBFNet neural network for point-wise
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

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized RBFNetPwiseClassif
        architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add RBFNetPwiseClassif's attributes to state dictionary
        state['num_classes'] = self.num_classes
        state['after_features_type'] = self.after_features_type
        state['after_features_dim'] = self.after_features_dim
        state['after_features_kernel_initializer'] = \
            self.after_features_kernel_initializer
        state['output_kernel_initializer'] = self.output_kernel_initializer
        state['binary_crossentropy'] = self.binary_crossentropy
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized RBFNetPwiseClassif architecture.

        :param state: The state's dictionary of the saved RBFNetPwiseClassif
            architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign RBFNetPwiseClassif's attributes from state dictionary
        self.num_classes = state['num_classes']
        self.after_features_type = state['after_features_type']
        self.after_features_dim = state['after_features_dim']
        self.after_features_kernel_initializer = \
            state['after_features_kernel_initializer']
        self.output_kernel_initializer = state['output_kernel_initializer']
        self.binary_crossentropy = state['binary_crossentropy']
        # Call parent's set state
        super().__setstate__(state)

    # ---  FIT LOGIC CALLBACKS  --- #
    # ----------------------------- #
    def prefit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately before
        fitting a RBFNetPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Prefit logic for RBF feature extraction layer representation
        if self.rbf_layers is not None:
            cache_map['Qpast'] = []
            for i, rbf_layer in enumerate(self.rbf_layers):
                rbf_layer.export_representation(
                    os.path.join(cache_map['rbf_dir_path'], f'RBF{i+1}_init'),
                    out_prefix=cache_map['out_prefix'],
                    Qpast=None
                )
                cache_map['Qpast'].append(np.array(rbf_layer.Q))

    def posfit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately after
        fitting a RBFNetPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Postfit logic for RBF feature extraction layer representation
        if self.rbf_layers is not None:
            for i, rbf_layer in enumerate(self.rbf_layers):
                rbf_layer.export_representation(
                    os.path.join(
                        cache_map['rbf_dir_path'], f'RBF{i+1}_trained'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Qpast=cache_map['Qpast'][i]
                )


