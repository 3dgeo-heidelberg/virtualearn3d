# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.initializer.kernel_point_ball_initializer import \
    KernelPointBallInitializer
from src.report.kpconv_layer_report import KPConvLayerReport
from src.plot.kpconv_layer_plot import KPConvLayerPlot
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class KPConvLayer(Layer):
    r"""

    :ivar sigma: The influence distance governing the kernel (:math:`\sigma`).
    :vartype sigma: float
    """
    # TODO Rethink : Doc
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        sigma=1.0,
        kernel_radius=1.0,
        num_kernel_points=19,
        deformable=False,
        Dout=320,
        built_Q=False,
        built_W=False,
        W_initializer=None,
        W_regularizer=None,
        W_constraint=None,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.sigma = sigma
        self.kernel_radius = kernel_radius
        self.num_kernel_points = num_kernel_points
        self.deformable = deformable
        self.Q_initializer = KernelPointBallInitializer(
            target_radius=self.kernel_radius,
            num_points=self.num_kernel_points,
            deformable=self.deformable,
            name='Q'
        )
        self.Dout = Dout
        self.W_initializer = tf.keras.initializers.get(W_initializer)
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        # Attributes initialized to None (derived when building)
        self.Q = None  # Kernel's structure matrix
        self.built_Q = built_Q  # True if built, False otherwise
        self.W = None  # Trainable kernel's weights
        self.built_W = built_W  # True if built, False otherwise

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        # TODO Rethink : Doc
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionality of the input feature space
        Din = dim_in[-2][-1]
        # Build the kernel's structure (if not yet)
        if not self.built_Q:
            self.Q = self.Q_initializer()
            self.built_Q = True
        # Validate the kernel's structure
        if self.Q is None:
            raise DeepLearningException(
                'KPConvLayer kernel structure was not built but it MUST be.'
            )
        if self.Q.shape[0] != self.num_kernel_points:
            raise DeepLearningException(
                f'KPConvLayer has {self.Q.shape[0]} kernel points but '
                f'{self.num_kernel_points} are expected.'
            )
        # Build the kernel's weights (if not yet)
        if not self.built_W:
            self.W = self.add_weight(
                shape=(self.num_kernel_points, Din, self.Dout),
                initializer=self.W_initializer,
                regularizer=self.W_regularizer,
                constraint=self.W_constraint,
                dtype='float32',
                trainable=True,
                name='W'
            )
        self.built = True

    def call(self, inputs, training=False, mask=False):
        # TODO Rethink : Doc
        # Extract input
        X = inputs[0]  # K x R x n_x=3
        F = inputs[1]  # K x R x n_1
        N = inputs[2]  # K x R x kappa
        # Gather neighborhoods (K x R x kappa x n_x=3 or n_d)
        NX, NF = self.gather_neighborhoods(N, X, F)
        # Return output features
        return self.compute_output_features(X, NX, NF)

    def gather_neighborhoods(self, N, X, F):
        # TODO Rethink : Doc
        NX = tf.gather(X, N, batch_dims=1, axis=1)
        NF = tf.gather(F, N, batch_dims=1, axis=1)
        return NX, NF

    def compute_output_features(self, X, NX, NF):
        # TODO Rethink : Doc
        # Compute correlated weights (K x R x kappa x m_q)
        Wc = NX-tf.expand_dims(X, axis=2)
        Wc = tf.tile(
            tf.expand_dims(Wc, axis=3),
            [1, 1, 1, self.num_kernel_points, 1]
        ) - self.Q
        Wc = 1-tf.sqrt(tf.reduce_sum(tf.square(Wc), axis=4))/self.sigma
        Wc = tf.maximum(0., Wc)
        # Compute weighted features
        WF = tf.transpose(
            tf.matmul(tf.transpose(Wc, [0, 1, 3, 2]), NF),
            [0, 2, 1, 3]
        )
        # Return output features
        return tf.reduce_sum(tf.matmul(WF, self.W), axis=1)

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'sigma': self.sigma,
            'kernel_radius': self.kernel_radius,
            'num_kernel_points': self.num_kernel_points,
            'deformable': self.deformable,
            'Dout': self.Dout,
            'W_initializer': tf.keras.initializers.serialize(
                self.W_initializer
            ),
            'W_regularizer': tf.keras.regularizers.serialize(
                self.W_regularizer
            ),
            'W_constraint': tf.keras.constraints.serialize(
                self.W_constraint
            )
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        kpcl = cls(**config)
        # Return deserialized layer
        return kpcl

    # ---   PLOTS and REPORTS   --- #
    # ----------------------------- #
    def export_representation(self, dir_path, out_prefix=None, Wpast=None):
        """
        Export a set of files representing the state of the kernel for both
        the structure (Q) and the weights (W).

        :param dir_path: The directory where the representation files will be
            exported.
        :type dir_path: str
        :param out_prefix: The output prefix to name the output files.
        :type out_prefix: str
        :param Wpast: The weights of the kernel in the past.
        :type Wpast: :class:`np.ndarray` or :class:`tf.Tensor` or None
        :return: Nothing at all, but the representation is exported as a set
            of files inside the given directory.
        """
        # Check dir_path has been given
        if dir_path is None:
            LOGGING.LOGGER.debug(
                'KPConvLayer.export_representation received no '
                'dir_path.'
            )
            return
        # Export the values (report) and the plots
        LOGGING.LOGGER.debug(
            'Exporting representation of KPConv layer to '
            f'"{dir_path}" ...'
        )
        start = time.perf_counter()
        # Export report
        KPConvLayerReport(
            np.array(self.Q), np.array(self.W)
        ).to_file(dir_path, out_prefix=out_prefix)
        # Export plots
        KPConvLayerPlot(
            Q=np.array(self.Q),
            W=np.array(self.W),
            Wpast=np.array(Wpast) if Wpast is not None else None,
            sigma=self.sigma,
            name=self.name,
            path=dir_path
        ).plot(out_prefix=out_prefix)
        # Log time
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            'Representation of KPConv layer exported to '
            f'"{dir_path}" in {end-start:.3f} seconds.'
        )
