# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.initializer.kernel_point_structure_initializer import\
    KernelPointStructureInitializer
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
    :vartype max_radii: :class:`np.ndarray` of float
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
    :ivar dim_out: The output dimensionality, it governs the trainable matrix
        :math:`\pmb{Q_W} \in \mathbb{R}^{n_f \times n_y}`.
    :vartype dim_out: int
    :ivar concatenation_strategy: Specify the concatenation strategy defining
        the output of the layer. It can be "FULL" so the generated output
        will be passed together with the input or "OPAQUE" in which case the
        input will not be forwarded as output.
    :vartype concatenation_strategy: str
    :ivar num_features: The number of features per point received as input
        :math:`n_f \in \mathbb{Z}_{>0}`.
    :vartype num_features: int
    :ivar concatf: The function implementing the concatenation strategy.
    :vartype concatf: func
    :ivar QX: The kernel's structure matrix.
    :vartype QX: :class:`tf.Tensor`
    :ivar trainable_QX: Flag to control whether QX is trainable or not.
    :vartype trainable_QX: bool
    :ivar built_QX: Flag to control whether QX has been built or not.
    :vartype built_QX: bool
    :ivar omegaD: The kernel's distance weights.
    :vartype omegaD: :class:`tf.Tensor`
    :ivar trainable_omegaD: Flag to control whether omegaD is trainable or not.
    :vartype trainable_omegaD: bool
    :ivar built_omegaD: Flag to control whether omegaD has been built or not.
    :vartype built_omegaD: bool
    :ivar omegaF: The kernel's feature weights.
    :vartype omegaF: :class:`tf.Tensor`
    :ivar trainable_omegaF: Flag to control whether omegaF is trainable or not.
    :vartype trainable_omegaF: bool
    :ivar built_omegaF: Flag to control whether omegaF has been built or not.
    :vartype built_omegaF: bool
    :ivar QW: The kernel's weights matrix.
    :vartype QW: :class:`tf.Tensor`
    :ivar trainable_QW: Flag to control whether QW is trainable or not.
    :vartype trainable_QW: bool
    :ivar built_QW: Flag to control whether QW has been built or not.
    :vartype built_QW: bool
    """
    # ---   INIT   --- #
    # ---------------- #
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
        self.trainable_QX = trainable_QX  # True is trainable, false not
        self.QX_initializer = KernelPointStructureInitializer(
            max_radii=self.max_radii,
            radii_resolution=self.radii_resolution,
            angular_resolutions=self.angular_resolutions,
            trainable=self.trainable_QX,
            name='QX'
        )
        self.num_kernel_points = \
            self.QX_initializer.compute_num_kernel_points()
        self.dim_out = dim_out
        self.concatenation_strategy = concatenation_strategy
        # Initialize to None attributes (derived when building)
        self.num_features = None  # The number of features
        self.concatf = None  # The concatenation strategy function
        self.QX = None  # Kernel's structure matrix
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
        Build the :math:`\pmb{Q_X} \in \mathbb{R}^{K \times n_x}` matrix
        representing the kernel's structure, and the
        :math:`\pmb{Q_W} \in \mathbb{R}^{n_f \times n_y}` matrix representing
        the kernel's weights as well as the
        :math:`\pmb{\omega_D} \in \mathbb{R}^{K}` and
        :math:`\pmb{\omega_F} \in \mathbb{R}^{n_f}` vectors.

        The :math:`\pmb{Q_X}` matrix represents the disposition of the
        kernel's points while the :math:`\pmb{Q_W}` is a matrix of weights
        that defines a potentially trainable transformation on the features.

        The :math:`\pmb{\omega_D}` vector governs the relevance
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
            self.QX = self.QX_initializer(None)
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
                name='QW'
            )
            self.built_QW = True
        # Handle concatenation strategy
        self.assign_concatf()

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the :math:`\pmb{Q_F} \in \mathbb{R}^{K \times n_f}`
        matrix and the output features :math:`\pmb{Q_Y}` obtained after
        matrix multiplication wrt :math:`\pmb{Q_W}`.

        Let the distance function between the point i of the kernel and the
        point j of the input point cloud be:

        .. math::
            d_Q(\pmb{q_{Xi*}}, \pmb{x_{j*}}) = \exp\left[
                - \dfrac{\lVert{ \pmb{x_{j*}} - \pmb{q_{Xi*}} }\rVert^2}{
                    \omega_{Di}^2
                }
            \right]

        .. math::
            \pmb{Q_F} \in \mathbb{R}^{k \times n_f} \;\text{ s.t. }\;
            q_{Fik} = \sum_{j=1}^{m}{
                d_Q(\pmb{q_{Xi*}}, \pmb{x_{j*}}) \omega_{Fk} f_{jk}
            }

        Alternatively, :math:`\pmb{Q_F}` rows can be expressed as a sum of
        vectors (where :math:`\odot` represents the Hadamard product):

        .. math::
            \pmb{q_{Fi*}} = \sum_{j=1}^{m}{
                d_Q(\pmb{q_{Xi*}}, \pmb{x_{j*}}) (
                    \pmb{\omega_F} \odot \pmb{f_{j*}}
                )
            }

        Note also that there exists a distance matrix
        :math:`\pmb{Q_D} \in \mathbb{R}^{k \times m}` describing how close each
        point from the input point cloud is wrt each kernel's point. It can be
        computed as:

        .. math::
            \pmb{Q_D} = \left[\begin{array}{ccc}
                q_{D11} & \ldots & q_{D1m} \\
                \vdots & \ddots & \vdots \\
                q_{Dk1} & \ldots & q_{Dkm}
            \end{array}\right]
            \; \text{ s.t. } \;
            Q_{Dij} = d_Q(\pmb{q_{Xi*}}, \pmb{x_{j*}})

        For then, the output matrix can be calculated as:

        .. math::
            \pmb{Q_Y} = \pmb{Q_D}^\intercal \pmb{Q_F} \pmb{Q_W}

        See :class:`.Layer` and :meth:`layer.Layer.call`.

        :return: The output features. Depending on the concatenation strategy
            they can be FULL :math:`[\pmb{F}, \pmb{Q_Y}]` or OPAQUE
            :math:`[\pmb{Q_Y}]`.
        """
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
        # Weight by kernel distances (NEW) ---
        """QD_row_wise_norm = tf.expand_dims(tf.reduce_sum(QD, axis=2), axis=-1)
        QD = QD / QD_row_wise_norm
        # Compute kernel-based features
        QF = self.omegaF * tf.matmul(QD, F)"""
        # --- Weight by kernel distances (NEW)
        # Weight by number of points (LEGACY) ---
        QF = self.omegaF/m * tf.matmul(QD, F)
        # --- Weight by number of points (LEGACY)
        # Multiply kernel features by weights
        QY = tf.matmul(QF, self.QW)
        QDT = tf.transpose(QD, [0, 2, 1])
        QY = tf.matmul(QDT, QY)
        # Return
        return self.concatf(F, QY)

    # ---   CALL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def concatf_full(F, QY):
        r"""
        :return: :math:`[\pmb{F}, \pmb{Q_Y}]`
        """

        return tf.concat([F, QY], axis=-1)

    @staticmethod
    def concatf_opaque(F, QY):
        r"""
        :return: :math:`[\pmb{Q_Y}]`
        """
        return QY

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def assign_concatf(self):
        """
        Assign the concatf function from the current state of the object.

        :return: Nothing, but self.concatf is updated.
        """
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

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's get_config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'max_radii': self.max_radii,
            'radii_resolution': self.radii_resolution,
            'angular_resolutions': self.angular_resolutions,
            'structure_dimensionality': self.structure_dimensionality,
            'num_kernel_points': self.num_kernel_points,
            'dim_out': self.dim_out,
            'concatenation_strategy': self.concatenation_strategy,
            # Building attributes
            'num_features': self.num_features,
            'trainable_QX': self.trainable_QX,
            'built_QX': self.built_QX,
            'trainable_omegaD': self.trainable_omegaD,
            'built_omegaD': self.built_omegaD,
            'trainable_omegaF': self.trainable_omegaF,
            'built_omegaF': self.built_omegaF,
            'trainable_QW': self.trainable_QW,
            'built_QW': self.built_QW
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Obtain num_kernel_points and remove it from config
        num_kernel_points = config['num_kernel_points']
        del config['num_kernel_points']
        # Obtain num_features and remove it from config
        num_features = config['num_features']
        del config['num_features']
        # Instantiate layer
        fsl = cls(**config)
        # Deserialize custom attributes
        fsl.num_features = num_features
        # Compute necessary initializations
        fsl.assign_concatf()
        # Placeholders so build on model load does not fail
        fsl.QX = tf.Variable(
            np.zeros((num_kernel_points, config['structure_dimensionality'])),
            dtype='float32',
            trainable=config['trainable_QX'],
            name='QX_placeholder'
        )
        fsl.omegaD = tf.Variable(
            np.zeros(num_kernel_points),
            dtype='float32',
            trainable=config['trainable_omegaD'],
            name='omegaD_placeholder'
        )
        fsl.omegaF = tf.Variable(
            np.zeros(num_features),
            dtype='float32',
            trainable=config['trainable_omegaF'],
            name='omegaF_placeholder'
        )
        fsl.QW = tf.Variable(
            np.zeros((num_features, config['dim_out'])),
            dtype='float32',
            trainable=config['trainable_QW'],
            name='QW_placeholder'
        )
        # Return deserialized layer
        return fsl

    # ---  PLOTS and REPORTS  --- #
    # --------------------------- #
    def export_representation(self, dir_path, out_prefix=None, QXpast=None):
        """
        Export a set of files representing the state of the features
        structuring kernel.

        :param dir_path: The directory where the representation files will
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
            QW=np.array(self.QW),
            xmax=np.max(self.max_radii),
            path=os.path.join(dir_path, 'figure.svg')
        ).plot(out_prefix=out_prefix)
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            'Representation of features structuring layer exported to '
            f'"{dir_path}" in {end-start:.3f} seconds.'
        )
