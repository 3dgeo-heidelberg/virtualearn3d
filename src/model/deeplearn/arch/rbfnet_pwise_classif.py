# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.arch.rbfnet import RBFNet
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.rbf_feat_extract_layer import \
    RBFFeatExtractLayer
import src.main.main_logger as LOGGING
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
        # Assign attributes
        self.num_classes = kwargs.get('num_classes', None)
        super().__init__(**kwargs)
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

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
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
        self.output_kernel_initializer = state['output_kernel_initializer']
        self.binary_crossentropy = state['binary_crossentropy']
        # Call parent's set state
        super().__setstate__(state)

