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
    :author: Alberto M. Esmoris Pena

    A kernel point convolution layer receives batches of :math:`R` points with
    :math:`D_{\mathrm{in}}` features each, and :math:`\kappa` known neighbors
    in the same space. These inputs are used to compute an output feature space
    of :math:`R` points with :math:`D_{\mathrm{out}}` features each. In doing
    so, the indexing tensor
    :math:`\mathcal{N} \in \mathbb{Z}^{K \times R \times \kappa}`
    is used to link :math:`\kappa` neighbors for each of the :math:`R` input
    points in each of the :math:`K` input batches.

    The layer is defined by a structure space matrix
    :math:`\pmb{Q} \in \mathbb{R}^{m_q \times n_x}` representing :math:`m_q`
    points in a :math:`n_x`-dimensional Euclidean space, and a tensor of
    weights :math:`\mathcal{W} \in \mathbb{R}^{m_q \times D_{\mathrm{in}} \times D_{\mathrm{out}}}`
    who has :math:`m_q` slices corresponding to a matrix of weights
    :math:`\pmb{W}_k \in \mathbb{R}^{D_{\mathrm{in}} \times D_{\mathrm{out}}}`
    associated to a different kernel point each.

    Further details on the KPConv operator can be read in the corresponding
    paper (https://arxiv.org/abs/1904.08889).


    :ivar sigma: The influence distance governing the kernel (:math:`\sigma`).
    :vartype sigma: float
    :ivar kernel_radius: The radius of the ball where the kernel is defined.
    :vartype kernel_radius: float
    :ivar num_kernel_points: How many points (:math:`m_q`) compose the kernel.
    :vartype num_kernel_points: int
    :ivar deformable: Whether the :math:`m_q` kernel points must be optimized
        (True) or not (False, def.) by the neural network.
    :vartype deformable: bool
    :ivar Dout: The output dimensionality of the convolution.
    :vartype Dout: int
    :ivar built_Q: Whether the structure space matrix
        :math:`\pmb{Q} \in \mathbb{R}^{m_q \times n_x}` is built. Initially it
        is false, but it will be updated once the layer is built.
    :vartype built_Q: bool
    :ivar built_W: Whether the :math:`m_q` weight matrices
        :math:`\pmb{W} \in \mathbb{D_{\mathrm{in}} \times D_{\mathrm{out}}}`
        corresponding to each kernel point are built or not.
        Initially it is false, but it will be updated once the
        layer is built.
    :vartype built_W: bool
    :ivar W_initializer: The initializer for the tensor of weights
        :math:`\mathcal{W} \in \mathbb{R}^{m_q \times D_{\mathrm{in}} \times D_{\mathrm{out}}}`.
    :ivar W_regularizer: The regularizer for the tensor of weights
        :math:`\mathcal{W} \in \mathbb{R}^{m_q \times D_{\mathrm{in}} \times D_{\mathrm{out}}}`.
    :ivar W_constraint: The constraints for the tensor of weights
        :math:`\mathcal{W} \in \mathbb{R}^{m_q \times D_{\mathrm{in}} \times D_{\mathrm{out}}}`.
    """
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
        r"""
        Build the :math:`\pmb{Q} \in \mathbb{R}^{m_q \times n_x}` matrix
        representing the kernel's structure space. Typically, the
        initialization is delegated to a
        :class:`.KernelPointBallInitializer` object. Also, build the
        :math:`\mathcal{W} \in \mathbb{R}^{m_q \times D_{\mathrm{in}} \times D_{\mathrm{out}}}`
        tensor of weights.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
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
        r"""
        Let :math:`\mathcal{Q} = (\pmb{Q}, \mathcal{W})` represent the
        kernel including both the structure and weights information.
        Also, let :math:`\pmb{P} = [\pmb{X} | \pmb{F}]` be the full
        matrix representation of the input receptive field including both the
        structure and the feature spaces.

        For then,
        the output for each receptive field in the input batch can be computed
        as a matrix :math:`\pmb{Y} \in \mathbb{R}^{R \times D_{\mathrm{out}}}`
        that arises by applying point-wise convolutions as shown below:


        .. math::
            \pmb{Y} = \left[\begin{array}{ccc}
                \rule[.5ex]{3.0ex}{0.6pt} &
                    (\pmb{P} * \mathcal{Q})(\pmb{x}_{1*}) &
                    \rule[.5ex]{3.0ex}{0.6pt} \\
                \vdots & \vdots & \vdots \\
                \rule[.5ex]{3.0ex}{0.6pt} &
                    (\pmb{P} * \mathcal{Q})(\pmb{x}_{R*}) &
                    \rule[.5ex]{3.0ex}{0.6pt}
            \end{array}\right]

        Each point-wise convolution is defined as:

        .. math::
            \left(\pmb{P} * \mathcal{Q}\right) =
                \sum_{\pmb{x}_{j*} \in \mathcal{N}_{\pmb{x}_{i*}}}{
                    \Biggl[{
                        \sum_{k=1}^{m_q} \max \; \biggl\{
                            0,
                            1 - \dfrac{
                                \lVert
                                    \pmb{x}_{j*} -
                                    \pmb{x}_{i*} -
                                    \pmb{q}_{k*}
                                \rVert
                            }{
                                \sigma
                            }
                        \biggr\}
                        }
                        \pmb{W}_{k}^\intercal
                    \Biggr]
                    \pmb{f}_{j*}
                }

        Where :math:`\mathcal{N}_{\pmb{x}_{i*}}` is the neighborhood of the
        :math:`\kappa` closest neighbors of :math:`\pmb{x}_{i*}` in the
        input receptive field :math:`\pmb{X}`.

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
                    \mathcal{N} \in \mathbb{Z}^{K \times R \times \kappa}


        :return: The output feature space
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times D_{\mathrm{out}}}`.
        """
        # Extract input
        X = inputs[0]  # K x R x n_x=3
        F = inputs[1]  # K x R x n_1
        N = inputs[2]  # K x R x kappa
        # Gather neighborhoods (K x R x kappa x n_x=3 or n_d)
        NX, NF = self.gather_neighborhoods(N, X, F)
        # Return output features
        return self.compute_output_features(X, NX, NF)

    def gather_neighborhoods(self, N, X, F):
        """
        Assist the :meth:`KPConvLayer.call` method by gathering the data
        representing the structure and feature spaces of each neighborhood.
        """
        NX = tf.gather(X, N, batch_dims=1, axis=1)
        NF = tf.gather(F, N, batch_dims=1, axis=1)
        return NX, NF

    def compute_output_features(self, X, NX, NF):
        """
        Assist the :meth:`KPConvLayer.call` method by applying the point-wise
        convolutions to compute the output features.
        """
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
