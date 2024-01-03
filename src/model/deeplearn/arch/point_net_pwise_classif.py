# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.features_structuring_layer import \
    FeaturesStructuringLayer
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import os


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
        self.final_shared_mlps = kwargs.get(
            'final_shared_mlps',
            [512, 256, 128]
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
        # Internal cache
        self.fsl_layer = None

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
        X, F = super().build_hidden(x, **kwargs), None
        if isinstance(X, list):
            X, F = X[0], X[1]
        # Extend parent's hidden layer with point-wise blocks
        X = tf.keras.layers.MaxPool1D(
            pool_size=self.num_points,
            name='max_pool1D_X'
        )(X)
        X = tf.tile(
            X,
            [1, self.num_points, 1],
            name='global_feats_X'
        )
        if F is not None:
            F = tf.keras.layers.MaxPool1D(
                pool_size=self.num_points,
                name='max_pool1D_F'
            )(F)
            F = tf.tile(
                F,
                [1, self.num_points, 1],
                name='global_feats_F'
            )
        # Concatenate features for point-wise classification
        x = []
        if self.skip_link_features_X:
            x = x + [self.Xtransf]
        if self.include_pretransf_feats_X:
            x = x + self.pretransf_feats_X
        if self.include_transf_feats_X:
            x = x + [self.transf_feats_X]
        if self.include_postransf_feats_X:
            x = x + [self.postransf_feats_X[:-1]]
        if self.include_global_feats_X:
            x = x + [X]
        if F is not None:
            if self.skip_link_features_F:
                x = x + [self.Ftransf]
            if self.include_pretransf_feats_F:
                x = x + self.pretransf_feats_F
            if self.include_transf_feats_F:
                x = x + [self.transf_feats_F]
            if self.include_postransf_feats_F:
                x = x + [self.postransf_feats_F[:-1]]
            if self.include_global_feats_F:
                x = x + [F]
        if len(x) < 1:
            raise DeepLearningException(
                'PointNetPwiseClassif cannot be built without features for '
                'point-wise classification.'
            )
        x = tf.keras.layers.Concatenate(name='full_feats')(x)
        # Final shared MLPs
        if self.final_shared_mlps is not None:
            for i, shared_mlp in enumerate(self.final_shared_mlps):
                x = PointNet.build_conv_block(
                    x,
                    filters=shared_mlp,
                    kernel_initializer=self.kernel_initializer,
                    name=f'final_sharedMLP{i+1}_{shared_mlp}'
                )
        # Convolve point-wise features
        if self.num_pwise_feats > 0:
            x = PointNet.build_conv_block(
                x,
                filters=self.num_pwise_feats,
                kernel_initializer=self.kernel_initializer,
                name='pwise_feats'
            )
        if self.features_structuring_layer is not None:
            x = self.build_features_structuring_layer(x)
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
                kernel_initializer=self.kernel_initializer,
                name='pwise_out'
            )(x)
        # Handle output layer for the general case
        return tf.keras.layers.Conv1D(
            self.num_classes,
            kernel_size=1,
            activation='softmax',
            kernel_initializer=self.kernel_initializer,
            name='pwise_out'
        )(x)

    # ---   FEATURES STRUCTURING LAYER   --- #
    # -------------------------------------- #
    def build_features_structuring_layer(self, x):
        """
        Build a features structuring layer to be computed on the features
        at given layer x.

        :param x: Given layer that has features as output.
        :type x: :class:`tf.Tensor`
        :return: The built features structuring layer.
        :rtype: :class:`.FeaturesStructuringLayer`
        """
        # Extract arguments to build the features structuring layer
        fsl = self.features_structuring_layer
        max_radii = fsl['max_radii']
        dim_out = fsl['dim_out']
        if max_radii == "AUTO":
            # TODO Rethink : Implement
            raise DeepLearningException(
                "AUTO max_radii for features structuring layer is not "
                "supported (yet)."
            )
        if dim_out == "AUTO":
            dim_out = self.num_pwise_feats
        # The features structuring layer itself
        self.fsl_layer = FeaturesStructuringLayer(
            max_radii=max_radii,
            radii_resolution=fsl['radii_resolution'],
            angular_resolutions=fsl['angular_resolutions'],
            concatenation_strategy=fsl['concatenation_strategy'],
            dim_out=dim_out,
            trainable_QX=fsl['trainable_kernel_structure'],
            trainable_QW=fsl['trainable_kernel_weights'],
            trainable_omegaD=fsl['trainable_distance_weights'],
            trainable_omegaF=fsl['trainable_feature_weights'],
            name='FSL'
        )
        x = self.fsl_layer([self.X, x])
        # Batch normalization
        if fsl['batch_normalization']:
            x = tf.keras.layers.BatchNormalization(
                momentum=0.0, name='FSL_BN'
            )(x)
        activation = fsl['activation']
        # Activation
        if activation is not None:
            activation_low = activation.lower()
            if activation_low == 'relu':
                x = tf.keras.layers.Activation(
                    "relu", name='FSL_RELU'
                )(x)
            else:
                raise DeepLearningException(
                    "Unexpected activation for features structuring layer:"
                    f" \"{activation}\""
                )
        return x

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized PointNetPwiseClassif
        architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add PointNetPwiseClassif's attributes to state dictionary
        state['num_classes'] = self.num_classes
        state['final_shared_mlps'] = self.final_shared_mlps
        state['num_pwise_feats'] = self.num_pwise_feats
        state['binary_crossentropy'] = self.binary_crossentropy
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized PointNetPwiseClassif architecture.

        :param state: The state's dictionary of the saved PointNetPwiseClassif
            architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign PointNetPwiseClassif's attributes from state dictionary
        self.num_classes = state['num_classes']
        self.final_shared_mlps = state['final_shared_mlps']
        self.num_pwise_feats = state['num_pwise_feats']
        self.binary_crossentropy = state['binary_crossentropy']
        self.fsl_layer = None
        # Call parent's set state
        super().__setstate__(state)

    # ---  FIT LOGIC CALLBACKS   --- #
    # ------------------------------ #
    def prefit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately before
        fitting a PointNetPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Prefit logic for features structuring layer representation
        if(
            self.fsl_layer is not None and
            cache_map.get('fsl_dir_path', None) is not None
        ):
            self.fsl_layer.export_representation(
                os.path.join(cache_map['fsl_dir_path'], 'init'),
                out_prefix=cache_map['out_prefix'],
                QXpast=None
            )
            cache_map['QXpast'] = np.array(self.fsl_layer.QX)
        # Prefit logic for freeze training
        if(
            self.features_structuring_layer is not None and
            self.features_structuring_layer.get('freeze_training', False)
        ):
            msg = 'Freeze training (prefit):\n'
            for layer in self.nn.layers[-4:-1]:
                msg += f'Disabled training for "{layer.name}".\n'
                layer.trainable = False
            LOGGING.LOGGER.debug(msg)
            cache_map['compilef'](cache_map['y_rf'])  # Recomp. to make effect.

    def posfit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately after
        fitting a PointNetPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Postfit logic for freeze training
        if (
            self.features_structuring_layer is not None and
            self.features_structuring_layer.get('freeze_training', False)
        ):
            # Prepare freeze training
            msg = 'Freeze training (posfit):\n'
            for layer in self.nn.layers:
                msg += f'Disabled training for "{layer.name}".\n'
                layer.trainable = False
            msg += '\n'
            for layer in self.nn.layers[-4:]:
                msg  += f'Enabled training for "{layer.name}".\n'
                layer.trainable = True
            LOGGING.LOGGER.debug(msg)
            cache_map['compilef'](cache_map['y_rf'])  # Recomp. to make effect.
            fsl_init_lr = self.features_structuring_layer.get(
                'freeze_training_init_learning_rate', None
            )
            if fsl_init_lr is not None:
                self.nn.optimizer.lr.assign(fsl_init_lr)
                if hasattr(self.nn.optimizer, '_learning_rate'):
                    _lr = self.nn.optimizer._learning_rate
                    if hasattr(_lr, 'initial_learning_rate'):
                        _lr.initial_learning_rate = fsl_init_lr
            # Fit unfrozen layers
            history = self.nn.fit(
                cache_map['X'], cache_map['y_rf'],
                epochs=cache_map['training_epochs'],
                callbacks=cache_map['callbacks'],
                batch_size=cache_map['batch_size']
            )
            # Merge history
            old_hist = cache_map['history']
            for key in old_hist.history.keys():
                old_hist.history[key] = np.concatenate(
                    (old_hist.history[key], history.history[key]),
                    axis=0
                )
            old_hist.params['epochs'] = (
                old_hist.params['epochs'] + history.params['epochs']
            )
            old_hist.params['steps'] = (
                old_hist.params['steps'] + history.params['steps']
            )
            old_hist.epoch += [
                1+i+int(np.max(old_hist.epoch)) for i in history.epoch
            ]
            cache_map['history'] = old_hist
            # Unfroze all layers
            msg = 'After freeze training (posfit):\n'
            for layer in self.nn.layers:
                msg += f'Enabled training for "{layer.name}".\n'
                layer.trainable = True
            msg += 'RECOMPILING IS NECESSARY FOR THESE CHANGES TO MAKE EFFECT!'
            LOGGING.LOGGER.debug(msg)
        # Postfit logic for features structuring layer representation
        if(
            self.fsl_layer is not None and
            cache_map.get('fsl_dir_path', None) is not None
        ):
            self.fsl_layer.export_representation(
                os.path.join(cache_map['fsl_dir_path'], 'trained'),
                out_prefix=cache_map['out_prefix'],
                QXpast=cache_map.get('QXpast', None)
            )

