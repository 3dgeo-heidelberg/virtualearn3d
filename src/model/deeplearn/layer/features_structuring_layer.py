# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.report.features_structuring_layer_report import \
    FeaturesStructuringLayerReport
from src.plot.features_structuring_layer_plot import \
    FeaturesStructuringLayerPlot
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import time
import os


# ---   CLASS   --- #
# ----------------- #
class FeaturesStructuringLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A features structuring layer is governed by a features structuring kernel.

    A features structuring kernel consists of a structure matrix
    :math:`\pmb{Q_X} \in \mathbb{R}^{K \times n_x}` for a structure space
    of dimensionality :math:`n_x` (typically 3D, i.e., :math:`x, y, z`),
    a features matrix :math:`\pmb{Q_F} \in \mathbb{R}^{K \times n_f}` and a
    kernel distance function
    :math:`d_Q(\pmb{q_x}, \pmb{x}) \in \mathbb{R}_{>0}`
    that quantifies the distance in the structure space between any kernel
    point wrt any input point.

    A features structuring kernel is generated from three main parameters.
    First,
    :math:`\pmb{r^*} \in \mathbb{R}^{n_x}`
    defines the radius along each axis
    for an ellipsoid, typically in 3D
    :math:`\pmb{r^*} = (r^*_x, r^*_y, r^*_z)`
    .
    Second, a radii resolution
    :math:`n \in \mathbb{Z}_{> 0}`
    defines how many concentric ellipsoids will be considered where the first
    one is the center point and the last one corresponds to the
    :math:`\pmb{r^*}`
    radii vector.
    Finally, :math:`m_1, \ldots, m_n` define the angular resolutions such that
    :math:`m_k \in \mathbb{Z}_{>0}` governs how many partitions along the
    angular domain of the :math:`\alpha` and :math:`\beta` angles must be
    considered.

    Any features structuring layer expects to receive two inputs in the
    following order:

    1) A structure space matrix representing a point cloud
    2) A matrix which rows are the point-wise features for the points in 1).

    :ivar max_radii: The radius of the last ellipsoid along each axis
        :math:`\pmb{r}^* \in \mathbb{R}^{n_x}`.
    :vartype: :class:`np.ndarray` of float
    :ivar radii_resolution: How many concentric ellipsoids must be considered
        :math:`n \in \mathbb{Z}_{>0}`
        (the first one is the center point, the last one is the biggest outer
        ellipsoid).
    :vartype radii_resolution: int
    :ivar angular_resolutions: How many angles consider for each ellipsoid
        :math:`(m_1, \ldots, m_n)`.
    :vartype angular_resolutions: :class:`np.ndarray` of int
    :ivar num_kernel_points: The number of points representing the kernel
        :math:`K \in \mathbb{Z}_{>0}`.
    :vartype num_kernel_points: int
    :ivar num_features: The number of features per point received as input
        :math:`n_f \in \mathbb{Z}_{>0}`.
    :vartype num_features: int
    """
    # ---   INIT   --- #
    # ----------------- #
    def __init__(
        self,
        max_radii,
        radii_resolution=4,
        angular_resolutions=(1, 2, 4, 8),
        structure_dimensionality=3,
        dim_out=4,
        concatenation_strategy='FULL',
        trainable_QX=False,
        trainable_QW=True,
        trainable_omegaD=True,
        trainable_omegaF=True,
        built_QX=False,
        built_omegaD=False,
        built_omegaF=False,
        built_QW=False,
        **kwargs
    ):
        """
        See :class:`.Layer` and
        :meth:`deeplearn.layer.layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.max_radii = np.array(max_radii)
        self.radii_resolution = int(radii_resolution)
        self.angular_resolutions = np.array(angular_resolutions, dtype=int)
        self.structure_dimensionality = structure_dimensionality
        self.num_kernel_points = int(np.sum(np.power(angular_resolutions, 2)))
        self.dim_out = dim_out
        self.concatenation_strategy = concatenation_strategy
        # Initialize to None attributes derived when building
        self.num_features = None  # The number of features
        self.concatf = None  # The concatenation strategy function
        self.QX = None  # Kernel's structure matrix
        self.trainable_QX = trainable_QX  # True is trainable, false not
        self.built_QX = built_QX  # True if built, false otherwise
        self.omegaD = None  # Trainable parameters for distance
        self.trainable_omegaD = trainable_omegaD  # True trainable, false not
        self.built_omegaD = built_omegaD  # True if built, false otherwise
        self.omegaF = None  # Trainable parameters for features
        self.trainable_omegaF = trainable_omegaF  # True trainable, false not
        self.built_omegaF = built_omegaF  # True if built, false otherwise
        self.QW = None  # Kernel's weights matrix
        self.trainable_QW = trainable_QW  # True is trainable, false not
        self.built_QW = built_QW  # True if built, false otherwise
        # Validate attributes
        if np.count_nonzero(self.max_radii <= 0):
            raise DeepLearningException(
                'FeaturesStructuringLayer does not support any max radii '
                'that is not strictly greater than zero.'
            )
        if len(self.angular_resolutions) != self.radii_resolution:
            raise DeepLearningException(
                'FeaturesStructuringLayer demands the cardinality of the '
                'angular resolutions set to match the radii resolution.'
            )
        if np.count_nonzero(self.angular_resolutions < 1):
            raise DeepLearningException(
                'FeaturesStructuringLayer demands all angular resolutions '
                'are strictly greater than zero.'
            )
        if self.structure_dimensionality != 3:
            raise DeepLearningException(
                'FeaturesStructuringLayer received '
                f'{self.structure_dimensionality} as structure dimensionality.'
                ' Currently, only 3D structure spaces are supported.'
            )
        if self.dim_out < 1:
            raise DeepLearningException(
                'FeaturesStructuringLayer received '
                f'{self.dim_out} as output dimensionality. '
                'But it must be greater than or equal to one.'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\pmb{Q_X}` matrix representing the kernel's structure
        as well as the :math:`\pmb{\omega_D} \in \mathbb{R}^{K}` and
        :math:`\pmb{\omega_F} \in \mathbb{R}^{n_f}` vectors of trainable
        parameters. The :math:`\pmb{\omega_D}` vector governs the relevance
        of the distance wrt to each kernel point when weighting the features,
        and the :math:`\pmb{\omega_F}` vectors governs the relevance of each
        feature independently of the distance in the structure space.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionalities
        Xdim, self.num_features = dim_in[0][-1], dim_in[1][-1]
        # Validate the dimensionalities
        if Xdim != self.structure_dimensionality:
            raise DeepLearningException(
                'FeaturesStructuringLayer received an input structure space '
                f'matrix representing an input point cloud in {Xdim} '
                'dimensions when the kernel\'s structure space has '
                f'{self.structure_dimensionality} dimensions.'
            )
        # Build the kernel's structure (if not yet)
        if not self.built_QX:
            self.QX = tf.Variable(
                self.sample_concentric_ellipsoids(),
                dtype='float32',
                trainable=self.trainable_QX,
                name='QX'
            )
            self.built_QX = True
        # Validate the kernel's structure
        if self.QX is None:
            raise DeepLearningException(
                'FeaturesStructuringLayer did NOT build any kernel\'s '
                'structure.'
            )
        if self.QX.shape[0] != self.num_kernel_points:
            raise DeepLearningException(
                'FeaturesStructuringLayer built a wrong kernel\'s structure '
                f'with {self.QX.shape[0]} points instead of '
                f'{self.num_kernel_points}.'
            )
        if self.QX.shape[1] != self.structure_dimensionality:
            raise DeepLearningException(
                'FeatureStructuringLayer built with a wrong structure '
                f'dimensionality of {self.QX.shape[1]}. It MUST be '
                f'{self.structure_dimensionality}.'
            )
        # Build the trainable weights (if not yet)
        if not self.built_omegaD:
            self.omegaD = tf.Variable(
                np.ones(self.num_kernel_points)*np.max(self.max_radii),
                dtype='float32',
                trainable=self.trainable_omegaD,
                name='omegaD'
            )
            self.built_omegaD = True
        if not self.built_omegaF:
            self.omegaF = tf.Variable(
                np.ones(self.num_features),
                dtype='float32',
                trainable=self.trainable_omegaF,
                name='omegaF'
            )
            self.built_omegaF = True
        if not self.built_QW:
            self.QW = tf.Variable(
                tf.keras.initializers.GlorotUniform()(
                    shape=(self.num_features, self.dim_out)
                ),
                dtype="float32",
                trainable=self.trainable_QW,
                name='"QW'
            )
            self.built_QW = True
        # Handle concatenation strategy
        cs_up = self.concatenation_strategy.upper()
        if cs_up == "FULL":
            self.concatf = self.concatf_full
        elif cs_up == "OPAQUE":
            self.concatf = self.concatf_opaque
        else:
            raise DeepLearningException(
                'FeaturesStructuringLayer received an unexpected '
                f'concatenation strategy "{self.concatenation_strategy}".'
            )

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the :math:`\pmb{Q_F} \in \mathbb{R}^{K \times n_f}`
        matrix and the output features derived by matrix multiplication.

        # TODO Rethink: Doc QF maths here

        See :class:`.Layer` and :meth:`layer.Layer.call`.

        :return: The output features. Depending on the concatenation strategy
            they can be FULL
            :math:`[\pmb{F}, \pmb{F}\pmb{Q_F}^\intercal, \pmb{F}\pmb{Q_F}^\intercal\pmb{Q_F}]`
            , KDIM
            :math:`[\pmb{F}, \pmb{F}\pmb{Q_F}^\intercal]`
            , FDIM
            :math:`[\pmb{F}, \pmb{F}\pmb{Q_F}^\intercal\pmb{Q_F}]`
            , FULL-OPAQUE
            :math:`[\pmb{F}\pmb{Q_F}^\intercal, \pmb{F}\pmb{Q_F}^\intercal\pmb{Q_F}]`
            , KDIM-OPAQUE
            :math:`[\pmb{F}\pmb{Q_F}^\intercal]`
            , FDIM-OPAQUE
            :math:`[\pmb{F}\pmb{Q_F}^\intercal\pmb{Q_F}]`
            .
        """
        # TODO Rethink : Validate layer
        # Extract input
        X = inputs[0]  # Input structure space matrix
        F = inputs[1]  # Input features matrix
        m = tf.cast(tf.shape(X)[-2], dtype="float32")  # Num input points
        # Compute the tensor of qx-x diffs for any qx in QX for any x in X
        SUBTRAHEND = tf.tile(
            tf.expand_dims(self.QX, 1),
            [1, tf.shape(X)[1], 1]
        )
        SUB = tf.subtract(tf.expand_dims(X, 1), SUBTRAHEND)
        # Compute kernel-pcloud distance matrix
        omegaD_squared = self.omegaD * self.omegaD
        QDunexp = tf.reduce_sum(SUB*SUB, axis=-1)
        QD = tf.exp(
            -tf.transpose(
                tf.transpose(QDunexp, [0, 2, 1]) / omegaD_squared, [0, 2, 1]
            )
        )

        # Compute kernel-based features
        QF = self.omegaF/m * tf.matmul(QD, F)
        # Multiply kernel features by weights
        QY = tf.matmul(QF, self.QW)
        QDT = tf.transpose(QD, [0, 2, 1])
        QY = tf.matmul(QDT, QY)
        # Return
        return self.concatf(F, QY)

    # ---   BUILD METHODS   --- #
    # ------------------------- #
    def sample_concentric_ellipsoids(self):
        """
        # TODO Rethink : Doc

        :return: :math:`\pmb{Q_X} \in \mathbb{R}^{K \times n_x}`
        """
        # TODO Rethink : Validate
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(3))
                continue
            # Concentric ellipsoid
            radii = k/(self.radii_resolution-1)*self.max_radii
            angular_resolution = self.angular_resolutions[k]
            for i in range(angular_resolution):
                alpha_i = i/angular_resolution*np.pi
                for j in range(angular_resolution):
                    beta_j = j/angular_resolution*2*np.pi
                    Q.append(np.array([
                        radii[0]*np.sin(alpha_i)*np.cos(beta_j),
                        radii[1]*np.sin(alpha_i)*np.sin(beta_j),
                        radii[2]*np.cos(alpha_i)
                    ]))
        # Return
        return np.array(Q)

    # ---   CALL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def concatf_full(F, QY):
        r"""
        :return: :math:`[\pmb{F}, \pmb{Q_F}, \pmb{Q_Y}]`
        """

        return tf.concat([F, QY], axis=-1)

    @staticmethod
    def concatf_opaque(F, QY):
        r"""
        :return: :math:`[\pmb{Q_Y}]`
        """
        return QY

    # TODO Rethink : Implement serialization

    # ---  PLOTS and REPORTS  --- #
    # --------------------------- #
    def export_representation(self, dir_path, out_prefix=None, QXpast=None):
        """
        Export a set of files representing the state of the features
        structuring kernel.

        :param dir_path: The directory where the representation's files will
            be exported.
        :type dir_path: str
        :param out_prefix: The output prefix to name the output files.
        :type out_prefix: str
        :param QXpast: The structure matrix of the layer in the past.
        :type QXpast: :class:`np.ndarray` or :class:`tf.Tensor` or None
        :return: Nothing at all, but the representation is exported as a set
            of files inside the given directory.
        """
        # Check dir_path has been given
        if dir_path is None:
            LOGGING.LOGGER.debug(
                'FeaturesStructuringLayer.export_representation received no '
                'dir_path.'
            )
            return
        # Export the values (report) and the plots
        LOGGING.LOGGER.debug(
            'Exporting representation of features structuring layer to '
            f'"{dir_path}" ...'
        )
        start = time.perf_counter()
        # Export report
        FeaturesStructuringLayerReport(
            np.array(self.QX), np.array(self.omegaF), np.array(self.omegaD),
            QXpast=np.array(QXpast) if QXpast is not None else None
        ).to_file(dir_path, out_prefix=out_prefix)
        # Export plots
        FeaturesStructuringLayerPlot(
            omegaD=np.array(self.omegaD),
            omegaF=np.array(self.omegaF),
            xmax=np.max(self.max_radii),
            path=os.path.join(dir_path, 'figure.svg')
        ).plot(out_prefix=out_prefix)
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            'Representation of features structuring layer exported to '
            f'"{dir_path}" in {end-start:.3f} seconds.'
        )
