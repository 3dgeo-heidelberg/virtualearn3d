# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.initializer.kernel_point_structure_initializer import\
    KernelPointStructureInitializer
import tensorflow as tf
import numpy as np


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
                \lVert{\pmb{x}_{i*}-\pmb{q}_{j*}}\rVert^2
            }{
                \omega_{j}^2
            }}\biggr)

    The :math:`\pmb{Y}` matrix can be understood as a feature matrix extracted
    from the :math:`m` input points through its interaction with the kernel.

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
    # TODO Rethink : Finish doc
    # TODO Rethink : Implement

    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        max_radii,
        radii_resolution=4,
        angular_resolutions=(1, 2, 4, 8),
        structure_dimensionality=3,
        trainable_Q=True,
        trainable_omega=True,
        built_Q=False,
        built_omega=True,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`deeplearn.layer.layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
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
            name='Q'
        )
        self.num_kernel_points = \
            self.Q_initializer.compute_num_kernel_points()
        self.trainable_omega = trainable_omega
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
        points while the :math:`\pmb{\omega}' vector represents the size of
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
                np.ones(self.num_kernel_points)*np.max(self.max_radii),
                dtype='float32',
                trainable=self.trainable_omega,
                name='omega'
            )
            self.built_omega = True

    def call(self, inputs, training=False, mask=False):
        r"""
        The computation of the :math:`\pmb{Y} \in \mathbb{R}^{m \times K}`
        matrix of output features.

        # TODO Rethink : Finish this doc with eqs? Otherwise write eqs in class doc


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
        omega_squared = self.omega * self.omega
        D_squared = tf.reduce_sum(SUB*SUB, axis=-1)
        # Compute the output features
        Y = tf.exp(
            -tf.transpose(D_squared, [0, 2, 1]) / omega_squared, [0, 1, 2]
        )
        # Return
        return Y

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
