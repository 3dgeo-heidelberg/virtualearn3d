# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class GroupingPointNetLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A grouping point net layer receives batches of :math:`R` points with
    :math:`n` features each (typically, :math:`n=n_x+n_f`, i.e., the sum of
    the input structure and feature spaces), and :math:`\kappa` known neighbors
    in the same space. These inputs are used to compute an output feature space
    of :math:`R` points with :math:`D_{\text{out}}` features each. In doing so,
    the indexing tensor
    :math:`\mathcal{N} \in \mathbb{Z}^{K \times R \times \kappa}`
    is used to link :math:`\kappa` neighbors for each of the :math:`R` input
    points in each of the :math:`K` input batches.

    The layer is defined by a matrix
    :math:`\pmb{H} \in \mathbb{R}^{D_{\text{out}} \times n}` that governs the
    weights for the SharedMLP (also, unitary 1DConv, i.e., win size one and
    stride one), and a pair (matrix, vector) governing the weights and the bias
    of the classical MLP (typical Dense layer in Keras):

    .. math::
        (
            \pmb{\Gamma} \in
                \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}},
            \pmb{\gamma} \in \mathbb{R}^{D_{\text{out}}}
        )

    The grouping PointNet layer applies a PointNet-like operator:

    .. math::
        \left(
            \gamma \circ \operatorname*{MAX}_{1 \leq i \leq \kappa}
        \right)\left(
            \left\{h(\pmb{p}_{i*})\right\}
        \right)

    To each of the neighborhoods, for each point in the batch, for each batch
    in the input. Where :math:`\pmb{p}_{i*} \in \mathbb{R}^{D_{\text{out}}}`
    is the vector representation of the :math:`i`-th point inside a given
    group (the neighborhood of a given input point). Note that
    :math:`\operatorname{MAX}` is the component-wise max, as explained in the
    PointNet paper (https://arxiv.org/abs/1612.00593).
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        dim_out,
        H_activation=None,
        H_initializer=None,
        H_regularizer=None,
        H_constraint=None,
        gamma_activation=None,
        gamma_kernel_initializer=None,
        gamma_kernel_regularizer=None,
        gamma_kernel_constraint=None,
        gamma_bias_enabled=True,
        gamma_bias_initializer=None,
        gamma_bias_regularizer=None,
        gamma_bias_constraint=None,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's
        super().__init__(**kwargs)
        # Assign attributes
        self.dim_out = dim_out
        self.H_activation = tf.keras.activations.get(H_activation)
        self.H_initializer = tf.keras.initializers.get(H_initializer)
        self.H_regularizer = tf.keras.regularizers.get(H_regularizer)
        self.H_constraint = tf.keras.constraints.get(H_constraint)
        self.gamma_activation = tf.keras.activations.get(gamma_activation)
        self.gamma_kernel_initializer = tf.keras.initializers.get(
            gamma_kernel_initializer
        )
        self.gamma_kernel_regularizer = tf.keras.regularizers.get(
            gamma_kernel_regularizer
        )
        self.gamma_kernel_constraint = tf.keras.constraints.get(
            gamma_kernel_constraint
        )
        self.gamma_bias_enabled = gamma_bias_enabled
        self.gamma_bias_initializer = tf.keras.initializers.get(
            gamma_bias_initializer
        )
        self.gamma_bias_regularizer = tf.keras.regularizers.get(
            gamma_bias_regularizer
        )
        self.gamma_bias_constraint = tf.keras.constraints.get(
            gamma_bias_constraint
        )
        # Build-initialized attributes
        self.feature_dimensionality = None
        self.structure_dimensionality = None
        self.H, self.gamma, self.gamma_bias = [None]*3
        # Validate attributes
        if self.dim_out is None or self.dim_out < 1:
            raise DeepLearningException(
                'GroupingPointNetLayer requires Dout > 0. '
                f'However, Dout = {self.dim_out} was given.'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\pmb{H} \in \mathbb{R}^{D_{\text{out}} \times n}`
        matrix representing the kernel or weights of the 1DConv or SharedMLP,
        also the
        :math:`\pmb{\Gamma} \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}`
        representing the weights of the MLP, and
        :math:`\pmb{\gamma} \in \mathbb{R}^{D_{\text{out}}}` representing its
        bias (if requested).


        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionalities
        self.structure_dimensionality = dim_in[0][-1]
        self.feature_dimensionality = dim_in[1][-1]
        # Build the coefficients for the h-functions
        self.H = self.add_weight(
            shape=(self.dim_out, self.calc_full_dimensionality()),
            initializer=self.H_initializer,
            regularizer=self.H_regularizer,
            constraint=self.H_constraint,
            dtype='float32',
            trainable=True,
            name='H_kernel'
        )
        # Build the coefficients for the gamma-functions
        self.gamma = self.add_weight(
            shape=(self.dim_out, self.dim_out),
            initializer=self.gamma_kernel_initializer,
            regularizer=self.gamma_kernel_regularizer,
            constraint=self.gamma_kernel_constraint,
            dtype='float32',
            trainable=True,
            name='gamma_kernel'
        )
        if self.gamma_bias_enabled:
            self.gamma_bias = self.add_weight(
                shape=(self.dim_out, ),
                initializer=self.gamma_bias_initializer,
                regularizer=self.gamma_bias_regularizer,
                constraint=self.gamma_bias_constraint,
                dtype='float32',
                trainable=True,
                name='gamma_bias'
            )
        self.built = True

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the
        :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times D_{\text{out}}}`
        output feature space.

        First, the structure and feature spaces are concatenated to compose the
        full point cloud matrix :math:`\pmb{P} = [\pmb{X} | \pmb{F}]` for each
        neighborhood in each receptive field in the batch. Then, these
        :math:`\pmb{P} \in \mathbb{R}^{\kappa \times n}` matrices are convolved
        through a SharedMLP (Unitary-1DConv) such that
        :math:`(\pmb{P}\pmb{H}^\intercal) \in \mathbb{R}^{\kappa \times D_{\text{out}}}`.

        Then, the component-wise max is computed to achieve a vector
        representation of each point
        :math:`\pmb{p}_{i*}^* \in \mathbb{R}^{D_{\text{out}}},\, i=1,\ldots,R`.
        These representations can be aranged as row-wise vectors in a matrix
        :math:`\pmb{P}^{*} \in \mathbb{R}^{R \times D_{\text{out}}}`.

        Finally, an MLP must be computed on the :math:`\pmb{P}^*` matrix. It
        can be done without bias:

        .. math::
            \pmb{Y} = \pmb{P}^{*} \pmb{\Gamma}

        Or it can be computed with bias:

        .. math::
            \pmb{Y} = (\pmb{P}^{*} \pmb{\Gamma}) \oplus \pmb{\gamma}

        Where :math:`\oplus` is the broadcast sum typical in machine learning
        contexts. More concretely, it is a sum of the
        :math:`\pmb{\gamma} \in \mathbb{R}^{D_{\text{out}}}` vector
        along the rows of the
        :math:`(\pmb{P}^* \pmb{\Gamma}) \in \mathbb{R}^{R \times D_{\text{out}}}`
        matrix.

        :param inputs: The input such that:

            -- inputs[0]
                is the structure space tensor representing the geometry of the
                many receptive fields in the batch.

                .. math::
                    \mathcal{X} \in \mathbb{R}^{K \times R \times n_x}

            -- inputs[1]
                is the feature space tensor representing the features of the
                many receptive fields in the batch.

                .. math::
                    \mathcal{F} \in \mathbb{R}^{K \times R \times n_f}

            -- inputs[2]
                is the indexing tensor representing the neighborhoods of
                :math:`\kappa` neighbors for each input point, in the same
                space.

                .. math::
                    \mathcal{N} \in \mathbb{R}^{K \times R \times \kappa}

        :return: The output feature space
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times D_{\text{out}}}`.
        """
        # Extract input
        X = inputs[0]
        F = inputs[1]
        N = inputs[2]
        # Concatenate X and F to have full dimensionality
        P = tf.concat([X, F], axis=2)  # P = [X | F]
        P_groups = tf.gather(P, N, axis=1, batch_dims=1)
        PHT = tf.tensordot(  # P x H^T (i.e., SharedMLP, Unitary-1DConv)
            P_groups,
            self.H,
            axes=[[3], [1]]
        )
        if self.H_activation is not None:
            PHT = self.H_activation(PHT)
        PHT_max = tf.reduce_max(  # Component-wise max on features vectors
            PHT,
            axis=2
        )  # K (batch size) x m (num points) x Dout (dim_out) x n (n_x+n_f)
        # Classic MLP (i.e., Dense) with weights in gamma
        gamma_max = tf.tensordot(PHT_max, self.gamma, axes=[[2], [0]])
        if self.gamma_bias_enabled:
            gamma_max = self.gamma_bias + gamma_max
        if self.gamma_activation is not None:
            gamma_max = self.gamma_activation(gamma_max)
        return gamma_max

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def calc_full_dimensionality(self):
        """
        Compute the full dimensionality on which the PointNet operator works.

        :return: The dimensionality of the feature space considered by the
            PointNet operator. Note it is not necessarily the same that
            ``self.feature_dimensionality`` because the structure space
            can be concatenated to the feature space before applying the
            PointNet operator.
        :rtype: int
        """
        return self.structure_dimensionality + self.feature_dimensionality

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's get_config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'dim_out': self.dim_out,
            'H_activation': tf.keras.activation.serialize(self.H_activation),
            'H_initializer': tf.keras.initializers.serialize(self.H_initializer),
            'H_regularizer': tf.keras.regularizers.serialize(self.H_regularizer),
            'H_constraint': tf.keras.constraints.serialize(self.H_constraint),
            'gamma_activation': tf.keras.activations.serialize(self.gamma_activation),
            'gamma_kernel_initializer': tf.keras.initializers.serialize(
                self.gamma_kernel_initializer
            ),
            'gamma_kernel_regularizer': tf.keras.regularizers.serialize(
                self.gamma_kernel_regularizer,
            ),
            'gamma_kernel_constraint': tf.keras.constraints.serialize(
                self.gamma_kernel_constraint
            ),
            'gamma_bias_enabled': self.gamma_bias_enabled,
            'gamma_bias_initializer': tf.keras.initializers.serialize(
                self.gamma_bias_initializer
            ),
            'gamma_bias_regularizer': tf.keras.regularizers.serialize(
                self.gamma_bias_regularizer
            ),
            'gamma_bias_constraint': tf.keras.constraints.serialize(
                self.gamma_bias_constraint
            ),
            # Build-initialized attributes
            'feature_dimensionality': self.feature_dimensionality,
            'structure_dimensionality': self.structure_dimensionality
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Obtain build-initialized attributes
        feature_dimensionality = config['feature_dimensionality']
        del config['feature_dimensionality']
        structure_dimensionality = config['structure_dimensionality']
        del config['structure_dimensionality']
        # Instantiate layer
        gpnl = cls(**config)
        # Assign build-initialized attributes
        gpnl.feature_dimensionality = feature_dimensionality
        gpnl.structure_dimensionality = structure_dimensionality
        # Return deserialized layer
        return gpnl
