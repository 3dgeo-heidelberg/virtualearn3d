# ---   IMPORTS   --- #
# ------------------- #
import time

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
import os


# ---   CLASS   --- #
# ----------------- #
class RBFFeatExtractLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A RBF feature extraction layer is governed by a matrix :math:`\pmb{Q}`
    representing the kernel's structure space in a :math:`n_x`-dimensional
    Euclidean space such that:

    .. math::

        \pmb{Q} \in \mathbb{R}^{K \times n_x}

    For 3D point clouds, :math:`n_x=3` and the matrix :math:`\pmb{Q}`
    represents the coordinates of the kernel points in a 3D space centered at
    zero.

    On top of that, a RBF feature extraction layer is also governed by a
    :math:`\pmb{\omega} \in \mathbb{R}^K` vector that parametrizes the
    curvature of each radial basis function.

    The output of a RBF feature extraction layer when considering a Gaussian
    kernel is given by the expression below:

    .. math::

        \pmb{Y} \in \mathbb{R}^{m \times K} \;\text{s.t.}\;
            y_{ij} = \exp\biggl({ - \dfrac{
                \lVert{\pmb{x_{i*}}-\pmb{q_{j*}}}\rVert^2
            }{
                \omega_{j}^2
            }}\biggr)

    The :math:`\pmb{Y}` matrix can be understood as a feature matrix extracted
    from the :math:`m` input points through its interaction with the kernel.

    :ivar structure_initialization_type: The type of initialization strategy
        for the kernel's structure space.
    :vartype structure_initialization_type: str
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
    :ivar kernel_function_type: The type of kernel function to be used
        (e.g., "Gaussian" or "Markov").
    :vartype kernel_function_type: str
    :ivar kernel_function: The kernel function to be used.
    :vartype kernel_function: function
    :ivar num_kernel_points: The number of points representing the kernel
        :math:`K \in \mathbb{Z}_{>0}`.
    :vartype num_kernel_points: int
    :ivar Q: The kernel's structure space matrix
    :vartype Q: :class:`tf.Tensor`
    :ivar trainable_Q: Flag to control whether Q is trainable or not.
    :vartype trainable_Q: bool
    :ivar built_Q: Flag to control whether Q has been built or not.
    :vartype built_Q: bool
    :ivar omega: The vector of kernel's sizes (can be thought as a curvature).
    :vartype omega: :class:`tf.Tensor`
    :ivar trainable_omega: Flag to control whether omega is trainable or not.
    :vartype trainable_omega: bool
    :ivar built_omega: Flag to control whether omega has been built or not.
    :vartype built_omega: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        max_radii,
        radii_resolution=4,
        angular_resolutions=(1, 2, 4, 8),
        kernel_function_type='Gaussian',
        structure_initialization_type='concentric_ellipsoids',
        structure_dimensionality=3,
        trainable_Q=True,
        trainable_omega=True,
        built_Q=False,
        built_omega=False,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.structure_initialization_type = structure_initialization_type
        self.max_radii = np.array(max_radii)
        self.radii_resolution = int(radii_resolution)
        self.angular_resolutions = np.array(angular_resolutions, dtype=int)
        self.structure_dimensionality = structure_dimensionality
        self.trainable_Q = trainable_Q  # True is trainable, false not
        self.Q_initializer = KernelPointStructureInitializer(
            max_radii=self.max_radii,
            radii_resolution=self.radii_resolution,
            angular_resolutions=self.angular_resolutions,
            trainable=self.trainable_Q,
            initialization_type=self.structure_initialization_type,
            name='Q'
        )
        self.num_kernel_points = \
            self.Q_initializer.compute_num_kernel_points()
        self.trainable_omega = trainable_omega
        # Handle kernel function type
        self.kernel_function_type = kernel_function_type
        kft_low = self.kernel_function_type.lower()
        if kft_low == 'gaussian':
            self.kernel_function = self.compute_gaussian_kernel
        elif kft_low == 'markov':
            self.kernel_function = self.compute_markov_kernel
        else:
            raise DeepLearningException(
                'RBFFeatExtractLayer does not support a kernel function of '
                f'type "{self.kernel_function_type}"'
            )
        # Initialize to None attributes (derived when building)
        self.Q = None  # Kernel's structure matrix
        self.built_Q = built_Q  # True if built, false otherwise
        self.omega = None  # Trainable kernel's size (think about curvature)
        self.built_omega = built_omega  # True if built, false otherwise
        # Validate attributes
        if self.structure_dimensionality != 3:
            raise DeepLearningException(
                'RBFFeatExtractLayer received '
                f'{self.structure_dimensionality} as structure dimensionality.'
                ' Currently, only 3D structure spaces are supported.'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\pmb{Q} \in \mathbb{K \times n_x}` matrix representing
        the kernel's structure, and the :math:`\pmb{\omega}` vector
        representing the kernel's sizes (think about curvature).

        The :math:`\pmb{Q}` matrix represents the disposition of the kernel's
        points while the :math:`\pmb{\omega}` vector represents the size of
        each radial basis function. The size of a radial basis function for
        typical Gaussian kernels can be seen as the parameter governing the
        curvature of the exponential.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionalities
        Xdim = dim_in[-1]
        # Validate the dimensionalities
        if Xdim != self.structure_dimensionality:
            raise DeepLearningException(
                'RBFFeatExtractLayer received an input structure space '
                f'matrix representing an input point cloud in {Xdim} '
                'dimensions when the kernel\'s structure space has '
                f'{self.structure_dimensionality} dimensions.'
            )
        # Build the kernel's structure (if not yet)
        if not self.built_Q:
            self.Q = self.Q_initializer(None)
            self.built_Q = True
        # Validate the kernel's structure
        if self.Q is None:
            raise DeepLearningException(
                'RBFFeatExtractLayer did NOT build any kernel\'s '
                'structure.'
            )
        if self.Q.shape[0] != self.num_kernel_points:
            raise DeepLearningException(
                'RBFFeatExtractLayer built a wrong kernel\'s structure '
                f'with {self.Q.shape[0]} points instead of '
                f'{self.num_kernel_points}.'
            )
        if self.Q.shape[1] != self.structure_dimensionality:
            raise DeepLearningException(
                'RBFFeatExtractLayer built with a wrong structure '
                f'dimensionality of {self.Q.shape[1]}. It MUST be '
                f'{self.structure_dimensionality}.'
            )
        # Build the kernel's sizes (if not yet)
        if not self.built_omega:
            self.omega = tf.Variable(
                np.max(self.max_radii) *
                    np.random.uniform(0.01, 1.0, self.num_kernel_points),
                dtype='float32',
                trainable=self.trainable_omega,
                name='omega'
            )
            self.built_omega = True

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the :math:`\pmb{Y} \in \mathbb{R}^{m \times K}`
        matrix of output features. For example, for a Gaussian kernel this
        matrix would be:

        .. math::

            y_{ij} = \exp\left(-\dfrac{
                \lVert{\pmb{x_{i*}} - \pmb{q_{j*}}}\rVert^2
            }{
                \omega_j^2
            }\right)

        Where :math:`\pmb{x_{i*}}` is the i-th input point and
        :math:`\pmb{q_{j*}}` is the j-th kernel point.

        :return: The extracted output features.
        :rtype: :class:`tf.Tensor`
        """
        # Extract input
        X = inputs  # Input structure space matrix
        # Compute the tensor of q-x diffs for any q in Q for any x in X
        SUBTRAHEND = tf.tile(
            tf.expand_dims(self.Q, 1),
            [1, tf.shape(X)[1], 1]
        )
        SUB = tf.subtract(tf.expand_dims(X, 1), SUBTRAHEND)
        # Compute kernel-pcloud distance matrix
        D_squared = tf.reduce_sum(SUB*SUB, axis=-1)
        # Compute and return output features
        return self.kernel_function(D_squared)

    # ---   KERNEL FUNCTIONS   --- #
    # ---------------------------- #
    def compute_gaussian_kernel(self, D_squared):
        r"""
        Compute a Gaussian kernel function.

        .. math::

            y_{ij} = \exp\left(
                - \dfrac{
                    \lVert{\pmb{x_{i*}}} - \pmb{q_{j*}}\rVert^2
                }{
                    \omega_{j}^2
                }
            \right)

        :return: The computed Gaussian kernel function.
        :rtype: :class:`tf.Tensor`
        """
        omega_squared = self.omega * self.omega
        return tf.exp(-tf.transpose(D_squared, [0, 2, 1]) / omega_squared)

    def compute_markov_kernel(self, D_squared):
        r"""
        Compute a Markov kernel function.

        .. math::

            y_{ij} = \exp\left(
                - \dfrac{
                    \lVert{\pmb{x_{i*}}} - \pmb{q_{j*}}\rVert
                }{
                    \omega_{j}^2
                }
            \right)

        :return: The computed Markov kernel function.
        :rtype: :class:`tf.Tensor`
        """
        omega_squared = self.omega * self.omega
        return tf.exp(-tf.transpose(D_squared, [0, 2, 1]) / omega_squared)

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's get_config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'structure_initialization_type': self.structure_initialization_type,
            'max_radii': self.max_radii,
            'radii_resolution': self.radii_resolution,
            'angular_resolutions': self.angular_resolutions,
            'kernel_function_type': self.kernel_function_type,
            'structure_dimensionality': self.structure_dimensionality,
            'num_kernel_points': self.num_kernel_points,
            # Building attributes
            'trainable_Q': self.trainable_Q,
            'built_Q': self.built_Q,
            'trainable_omega': self.trainable_omega,
            'built_omega': self.built_omega
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Obtain num_kernel_points and remove it from config
        num_kernel_points = config['num_kernel_points']
        del config['num_kernel_points']
        # Instantiate layer
        rfel = cls(**config)
        # Placeholders so build on model load does not fail
        rfel.Q = tf.Variable(
            np.zeros((num_kernel_points, config['structure_dimensionality'])),
            dtype='float32',
            trainable=config['trainable_Q'],
            name='Q_placeholder'
        )
        rfel.omega = tf.Variable(
            np.zeros(num_kernel_points),
            dtype='float32',
            trainable=config['trainable_omega'],
            name='omega_placeholder'
        )
        # Return deserialized layer
        return rfel

    # ---  PLOTS and REPORTS  --- #
    # --------------------------- #
    def export_representation(self, dir_path, out_prefix=None, Qpast=None):
        """
        Export a set of files representing the state of the kernel's structure.

        :param dir_path: The directory where the representation files will be
            exported.
        :type dir_path: str
        :param out_prefix: The output prefix to name the output files.
        :type out_prefix: str
        :param Qpast: The structure matrix of the layer in the past.
        :type Qpast: :class:`np.ndarray` or :class:`tf.Tensor` or None
        :return: Nothing at all, but the representation is exported as a set
            of files inside the given directory.
        """
        # Check dir_path has been given
        if dir_path is None:
            LOGGING.LOGGER.debug(
                'RBFFeatExtractLayer.export_representation received no '
                'dir_path.'
            )
            return
        # Export the values (report) and the plots
        LOGGING.LOGGER.debug(
            'Exporting representation of RBF feature extraction layer to '
            f'"{dir_path}" ...'
        )
        start = time.perf_counter()
        # Export report
        FeaturesStructuringLayerReport(
            np.array(self.Q), None, np.array(self.omega),
            QXpast=np.array(Qpast) if Qpast is not None else None,
            QX_name='Q',
            omegaD_name='omega'
        ).to_file(dir_path, out_prefix=out_prefix)
        # Export plots
        FeaturesStructuringLayerPlot(
            omegaD=np.array(self.omega),
            xmax=np.max(self.max_radii),
            path=os.path.join(dir_path, 'figure.svg'),
            omegaD_name='$\\omega_{i}$',
            kernel_type=self.kernel_function_type
        ).plot(out_prefix=out_prefix)
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            'Representation of RBF feature extraction layer exported to '
            f'"{dir_path}" in {end-start:.3f} seconds.'
        )
