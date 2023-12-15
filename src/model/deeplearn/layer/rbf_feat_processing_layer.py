# ---   IMPORTS   --- #
# ⨪------------------ #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
# TODO Rethink : Implement ---
"""from src.report.feature_processing_layer_report import \
    FeatureProcecssingLayerReport
from src.plot.feature_processing_layer_plot import \
    FeatureProcessingLayerPlot"""
# --- TODO Rethink : Implement
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import time
import os


# ---   CLASS   --- #
# ----------------- #
class RBFFeatProcessingLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A RBF feature processing layer is governed by a matrix
    :math:`\pmb{M} \in \mathbb{R}^{K \times n_f}`
    representing the :math:`K` kernels for each of the :math:`n_f` features,
    and a matrix :math:`\pmb{\Omega} \in \mathbb{R}^{K \times n_f}`.

    Each column :math:`\pmb{\mu}_{*k}` of the matrix :math:`\pmb{M}` defines
    :math:`K` kernels for a given input feature, together with each column
    :math:`\pmb{\omega}_{*k}` of the matrix :math:`\pmb{\Omega}`.

    The output of a RBFFeatProcessingLayer consists of a matrix
    :math:`\pmb{Y}\in\mathbb{R}^{m \times K n_f}` with :math:`K \times n_f`
    output features for each of the :math:`m` input points.

    Let :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}` be a tensor
    that can be sliced into :math:`K` matrices of :math:`m` rows and
    :math:`n_f` columns representing the point-wise output features derived
    from a given kernel. For then, any cell of this tensor can be defined
    as follows (assuming a Gaussian RBF):

    .. math::

        \mathcal{y}_{kij} = \exp\left[-\dfrac{
            (f_{ij} - \mu_{kj})^2
        }{
            \omega_{kj}^2
        }\right]

    Now, **the output matrix** :math:`\pmb{Y}` can be simply defined by
    reorganizing the tensor :math:`\mathcal{Y}` as a matrix such that:

    .. math::

        \pmb{Y} = \left[\begin{array}{ccccc}
            | & & | & & | \\
            \pmb{y}_{1*1} & \cdots & \pmb{y}_{K*1} & \cdots & \pmb{y}_{K*n_f} \\
            | & & | & & |
        \end{array}\right]

    **Technically**, it is also convenient to express the output matrix like a
    component-wise exponential
    :math:`\pmb{Y} = \exp\left[- \pmb{D} \odot \pmb{D}\right]`, where
    :math:`\odot` is the Hadamard product and
    :math:`\pmb{D} \in \mathbb{R}^{m \times K n_f}` is a matrix of
    scaled differences as defined below:

    .. math::

        \pmb{D} = \left[\begin{array}{ccccc}
            | & & | & & | \\
            \dfrac{\pmb{f}_{*1}-\mu_{11}}{\omega_{11}} & \cdots &
                \dfrac{\pmb{f}_{*1}-\mu_{K1}}{\omega_{K1}} & \cdots &
                \dfrac{\pmb{f}_{*n_f}-\mu_{Kn_f}}{\omega_{Kn_f}} \\
            | & & | & & |
        \end{array}\right]

    For **initialization**, the mean for each :math:`j`-th feature
    :math:`\mu_j` is assumed, together with its standard deviation
    :math:`\sigma_j`. With this information, it is possible to initialize
    the columns of the matrix :math:`\pmb{M}` by taking :math:`K`
    linearly-spaced samples from the interval
    :math:`[\mu_j - 3\sigma_j, \mu_j + 3\sigma_j]`. Besides, the rows of the
    matrix :math:`\pmb{\Omega}` can be initialized considering samples from a
    uniform distribution :math:`x \sim U(-b, b)` such that
    :math:`\forall 1 < k \leq K,\, \omega_{kj} = a + \sigma_j(b+x)`. Typically,
    :math:`a=10^{-2}` and :math:`b=1`.

    TODO Rethink : Doc ivars and vartypes
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        num_kernels,
        means,
        stdevs,
        a=0.01,
        b=1,
        kernel_function_type='Gaussian',
        trainable_M=True,
        trainable_Omega=True,
        built_M=False,
        built_Omega=False,
        **kwargs
    ):
        """
        See :class:`.Layer` :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.means = np.array(means) # Each mu_j, j=1,...,n_f
        self.stdevs = np.array(stdevs) # Each sigma_j, j=1,...,n_f
        self.num_feats = len(self.means)  # n_f
        self.num_kernels = num_kernels  # K
        self.a = a  # a
        self.b = b  # b
        self.trainable_M = trainable_M
        self.trainable_Omega = trainable_Omega
        # Handle kernel function type
        self.kernel_function_type = kernel_function_type
        kft_low = self.kernel_function_type.lower()
        if kft_low == 'gaussian':
            self.kernel_function = self.compute_gaussian_kernel
        elif kft_low == 'markov':
            self.kernel_function = self.compute_markov_kernel
        else:
            raise DeepLearningException(
                'RBFFeatProcessingLayer does not support a kernel function of '
                f'type "{self.kernel_function_type}".'
            )
        # Initialize to None attributes (derived when building)
        self.M = None  # Kernel's matrix of centers
        self.built_M = built_M  # True if built, false otherwise
        self.Omega = None  # Kernel's matrix of curvatures (AKA kernel's sizes)
        self.built_Omega = built_Omega  # True if built, false otherwise
        # Validate attributes
        if len(self.means) != len(self.stdevs):
            raise DeepLearningException(
                f'RBFFeatProcessingLayer received {len(self.means)} means and '
                f'{len(self.stdevs)} standard deviations. '
                'Those numbers are different but they MUST be the same.'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\pmb{M} \in \mathbb{R}^{K \times n_f}` and
        :math:`\pmb{\Omega} \in \mathbb{R}^{K \times n_f}` matrices
        representing the feature processing kernel's centers and sizes
        (curvatures), respectively.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's biuld
        super().build(dim_in)
        # Find the dimensionality
        nf = dim_in[-1]  # Number of features
        # Validate the dimensionality
        if nf != self.num_feats:
            raise DeepLearningException(
                'RBFFeatProcessingLayer received an input feature matrix '
                f'with {nf} features (columns) but {self.num_feats} were '
                'expected.'
            )
        # Build the kernel's centers (if not yet)
        if not self.built_M:
            self.M = tf.Variable(
                np.linspace(
                    self.means-3*self.stdevs,
                    self.means+3*self.stdevs,
                    self.num_kernels
                ),
                dtype='float32',
                trainable=self.trainable_M,
                name='M'
            )
            self.built_M = True
        # Build the kernel's sizes (if not yet)
        if not self.built_Omega:
            self.Omega = tf.Variable(
                np.array([self.a + self.stdevs[j] * (
                    self.b + np.random.uniform(
                        -self.b, self.b, self.num_kernels
                    )
                ) for j in range(self.num_feats)]).T,
                dtype='float32',
                trainable=self.trainable_Omega,
                name='Omega'
            )

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the :math:`\pmb{Y} \in \mathbb{m \times Kn_f}`
        output matrix.

        :return: The processed output features.
        :rtype: :class:`tf.Tensor`
        """
        # Extract input
        F = inputs  # Input feature space matrix
        # Compute and return output features
        return self.kernel_function(F)

    # ---   KERNEL FUNCTIONS   --- #
    # ---------------------------- #
    def compute_gaussian_kernel(self, F):
        r"""
        Compute a Gaussian kernel function.

        .. math::

            y_{ip} = \exp\left[
                - \dfrac{(f_{ij} - \mu_{kj})^2}{\omega_{kj}^2}
            \right]

        :param F: The feature space matrix.
        :return: The computed Gaussian kernel function.
        """
        # Compute the matrix of scaled differences
        F_repeated = tf.repeat(
            F,
            self.num_kernels,  # Repeat for each kernel
            axis=2  # Repeat along feature axis
        )
        M = tf.reshape(tf.transpose(self.M), shape=[-1])
        Omega = tf.reshape(tf.transpose(self.Omega), shape=[-1])
        D = (F_repeated-M) / Omega
        # Return the Gaussian RBF
        return tf.exp(-tf.square(D))

    def compute_markov_kernel(self, F):
        r"""
        Compute a Markov kernel function.

        .. math::

            y_{ip} = \exp\left[
                - \dfrac{\lvert{f_{ij} - \mu_{kj}}\rvert}{\omega_{kj}^2}
            \right]

        :param F: The feature space matrix.
        :return: The computed Markov kernel function.
        """
        # Compute the matrix of scaled differences
        F_repeated = tf.repeat(
            F,
            self.num_kernels,  # Repeat for each kernel
            axis=2  # Repeat along feature axis
        )
        M = tf.reshape(tf.transpose(self.M), shape=-1)
        Omega = tf.reshape(tf.transpose(self.Omega), shape=-1)
        # Return the Gaussian RBF
        return tf.exp(-tf.abs(F_repeated-M)/tf.square(Omega))


    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's get_config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'means': self.means,
            'stdevs': self.stdevs,
            'num_kernels': self.num_kernels,
            'a': self.a,
            'b': self.b,
            'kernel_function_type': self.kernel_function_type,
            # Building attributes
            'trainable_M': self.trainable_M,
            'built_M': self.built_M,
            'trainable_Omega': self.trainable_Omega,
            'built_Omega': self.built_Omega
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Extract values of interest
        rfpl = cls(**config)
        num_kernels = config['num_kernels']
        num_feats = len(config['means'])
        # Placeholders so build on model load does not fail
        rfpl.M = tf.Variable(
            np.zeros((num_kernels, num_feats)),
            dtype='float32',
            trainable=config['trainable_M'],
            name='M_placeholder'
        )
        rfpl.Omega = tf.Variable(
            np.zeros((num_kernels, num_feats)),
            dtype='float32',
            trainable=config['trainable_Omega'],
            name='Omega_placeholder'
        )
        # Return deserialized layer
        return rfpl

    # TODO Rethink : Plots and reports (see RBFFeatExtractLayer)