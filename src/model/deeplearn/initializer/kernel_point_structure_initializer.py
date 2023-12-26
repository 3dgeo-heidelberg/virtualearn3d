# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.initializer.initializer import Initializer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class KernelPointStructureInitializer(Initializer):
    r"""
    :author: Alberto M. Esmoris Pena

    A kernel point structure initializer initializes the structure space matrix
    representing the support points of a kernel. It will generate a structure
    space matrix for the kernel :math:`\pmb{Q} \in \mathbb{R}^{K \times n_x}`.
    This matrix has :math:`K` rows representing :math:`K` points in a
    :math:`n_x`-dimensional space, typically 3D, i.e., :math:`n_x=3`.

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
    :ivar trainable: Flag to control whether :math:`\pmb{Q}`  is trainable or
        not.
    :vartype trainable: bool
    :ivar name: The name for the initialized tensor.
    :vartype name: str
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        initialization_type='concentric_ellipsoids',
        max_radii=(1, 1, 1),
        radii_resolution=4,
        angular_resolutions=(1, 2, 4, 8),
        trainable=True,
        name='Q',
        **kwargs
    ):
        """
        Initialize the member attributes of the initializer.

        :param kwargs: The key-word specification to parametrize the
            initializer.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.initialization_type = initialization_type
        self.max_radii = np.array(max_radii)
        self.radii_resolution = int(radii_resolution)
        self.angular_resolutions = np.array(angular_resolutions, dtype=int)
        self.trainable = trainable
        self.name = name
        # Validate attributes
        if np.count_nonzero(self.max_radii <= 0):
            raise DeepLearningException(
                'KernelPointStructureInitializer does not support any max '
                'radii that is not strictly greater than zero.'
            )
        if len(self.angular_resolutions) != self.radii_resolution:
            raise DeepLearningException(
                'KernelPointStructureInitializer demands the cardinality of '
                'the angular resolutions set to match the radii resolution.'
            )
        if np.count_nonzero(self.angular_resolutions < 1):
            raise DeepLearningException(
                'KernelPointStructureInitializer demands all angular '
                'resolutions are strictly greater than zero.'
            )

    # ---   INITIALIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, shape=None, dtype='float32'):
        r"""
        Initialize a structure space matrix representing the support points
        of a kernel.

        :param shape: The shape parameter is ignored as the dimensionality
            of the initialized matrix depends on the arguments used to build
            the initializer.
        :return: The structure space matrix for a point-based kernel
            :math:`\pmb{Q} \in \mathbb{R}^{K \times n_x}`.
        :rtype: :class:`tf.Tensor`
        """
        # Prepare initialization
        init_type = self.initialization_type.lower()
        num_kernel_points = self.compute_num_kernel_points()
        structure_dim = 3
        Q = None
        # Initialize kernel structure
        if init_type == 'concentric_ellipsoids':
            Q = self.sample_concentric_ellipsoids()
        elif init_type == 'concentric_rectangulars':
            Q = self.sample_concentric_rectangulars()
        elif init_type == 'concentric_grids':
            Q = self.sample_concentric_grids()
        elif init_type == 'concentric_cylinders':
            Q = self.sample_concentric_cylinders()
        elif init_type == 'cone':
            Q = self.sample_cone()
        elif init_type == 'zeros':
            Q = np.zeros((num_kernel_points, structure_dim))
        # Handle unexpected initialization type
        else:
            raise DeepLearningException(
                'KernelPointStructureInitializer received an unexpected '
                f'initialization type: "{self.initialization_type}"'
            )
        # Return initialized kernel structure as tensor
        return tf.Variable(
            Q,
            dtype=dtype,
            trainable=self.trainable,
            name=self.name
        )

    # ---   BUILD METHODS   --- #
    # ------------------------- #
    def sample_concentric_ellipsoids(self):
        r"""
        The structure matrix of the kernel :math:`\pmb{Q} \in \mathbb{R}^{3}`
        is initialized for the 3D case assuming three parameters:

        :math:`n` The radii resolution.

        :math:`r_x^*, r_y^*, r_z^*` The max radii.

        :math:`m_1, \ldots, m_n` The angular resolution for each radius.

        First, the axis-wise radii for given ellipsoid are defined as:

        .. math::
            r_{xk} = \frac{k-1}{n-1} r_x^* \\
            r_{yk} = \frac{k-1}{n-1} r_y^* \\
            r_{zk} = \frac{k-1}{n-1} r_z^*

        Where :math:`k=0,...,n-1` represents the :math:`n` concentric
        ellipsoids with :math:`k=0` being the central point and
        :math:`k=n-1` the biggest ellipsoid, i.e., from smaller to bigger.

        For then, it is possible to define two angles
        :math:`\alpha \in [0, \pi]` and :math:`\beta \in [0, 2\pi]` such that:

        .. math::
            \alpha_{kj} = \frac{j-1}{m_k - 1} \pi \;,\;\;
            \beta_{kj} = \frac{j-1}{m_k -1} 2\pi

        Where :math:`j=0, \ldots, m_k-1` for each :math:`k`.

        Finally, the rows in :math:`\pmb{Q}` that represent the kernel's
        structure can be computed as follows:

        .. math::
            \left[\begin{array}{c}
                x \\
                y \\
                z
            \end{array}\right]^\intercal =
            \left[\begin{array}{c}
                r_{xk} \sin(\alpha_{kj}) \cos(\beta_{kj}) \\
                r_{yk} \sin(\alpha_{kj}) \sin(\beta_{kj}) \\
                r_{zk} \cos(\alpha_{kj})
            \end{array}\right]^\intercal

        :return: :math:`\pmb{Q} \in \mathbb{R}^{K \times n_x}`
        :rtype: :class:`tf.Tensor`
        """
        # Generate kernel's structure
        dimensions=3
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(dimensions))
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
        Q = np.array(Q)
        # Validate
        if Q.shape[0] != self.compute_num_kernel_points():
            raise DeepLearningException(
                'KernelPointStructureInitializer generated an unexpected '
                'number of kernel points for concentric ellipsoids.'
            )
        # Return
        return Q

    def sample_concentric_grids(self):
        r"""
        In the concentric grids strategy the angular resolution
        is understood as the edge resolution :math:`r_e`.

        One grid will be generated for each radius in the radii
        resolution specification. More concretely, each grid will
        have a length side of twice the radius and a number of points equal
        to :math:`r_e^3` for the 3D case. Note that a radius is given for each
        axis such that rectangular grids are supported in general, not only
        voxel grids (those correspond to all axis having the same radius).

        :return: Points sampled from concentric grids.
        :rtype: :class:`tf.Tensor`
        """
        # Generate kernel's structure
        dimensions = 3
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(3))
                continue
            # Concentric grids
            radii = k/(self.radii_resolution-1)*self.max_radii
            edge_resolution = self.angular_resolutions[k]
            tx = np.linspace(-radii[0], radii[0], edge_resolution)
            ty = np.linspace(-radii[1], radii[1], edge_resolution)
            tz = np.linspace(-radii[2], radii[2], edge_resolution)
            for x in tx:
                for y in ty:
                    for z in tz:
                        Q.append(np.array([x, y, z]))
        Q = np.array(Q)
        # Validate
        if Q.shape[0] != self.compute_num_kernel_points():
            raise DeepLearningException(
                'KernelPointStructureInitializer generated an unexpected '
                'number of kernel points for concentric grids'
            )
        # Return
        return Q

    def sample_concentric_rectangulars(self):
        r"""
        In the concentric rectangular prisms strategy the angular resolution
        is understood as the edge resolution :math:`r_e`.

        One rectangular prism will be generated for each radius in the radii
        resolution specification. More concretely, each rectangular prism will
        have a length side of twice the radius and a number of points equal
        to :math:`6r_e^2-12r_3+8` for the 3D case. Note that a radius is given
        for each axis such that rectangular prisms are supported in general,
        not only voxels (those correspond to all axis having the same radius).

        :return: Points sampled from concentric rectangular prisms.
        :rtype: :class:`tf.Tensor`
        """
        # Generate kernel's structure
        dimensions = 3
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(3))
                continue
            # Concentric rectangular prisms
            radii = k/(self.radii_resolution-1)*self.max_radii
            edge_resolution = self.angular_resolutions[k]
            tx = np.linspace(-radii[0], radii[0], edge_resolution)
            ty = np.linspace(-radii[1], radii[1], edge_resolution)
            tz = np.linspace(-radii[2], radii[2], edge_resolution)
            # Find x,y boundary
            for x in [tx[0], tx[-1]]:  # For each extreme x
                for y in ty:  # For each y
                    for z in tz:  # For each z
                        Q.append(np.array([x, y, z]))
            for y in [ty[0], ty[-1]]:  # For each extreme y
                for x in tx[1:-1]:  # For each non-extreme x
                    for z in tz:  # For each z
                        Q.append(np.array([x, y, z]))
            for x in tx[1:-1]:  # For each non-extreme x
                for y in ty[1:-1]:  # For each non-extreme y
                    for z in [tz[0], tz[-1]]:  # For each extreme z
                        Q.append(np.array([x, y, z]))
        Q = np.array(Q)
        # Validate
        if Q.shape[0] != self.compute_num_kernel_points():
            raise DeepLearningException(
                'KernelPointStructureInitializer generated an unexpected '
                'number of kernel points for concentric grids'
            )
        # Return
        return Q

    def sample_concentric_cylinders(self):
        r"""
        In the concentric cylinders strategy the angular resolution
        :math:`r_a` is used directly to sample along the :math:`z`-axis.

        One cylinder will be generated for each radius in the radii
        resolution specification. More concretely, each cylinder will
        have a number of points equal to :math:`r_a^2r_r` where
        :math:`r_r` is the radius of the disk for the current cylinder.

        :return: Points sampled from concentric cylinders.
        :rtype: :class:`tf.Tensor`
        """
        # Generate kernel's structure
        dimensions = 3
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(3))
                continue
            # Concentric cylinders
            radii = k/(self.radii_resolution-1)*self.max_radii
            angular_resolution = self.angular_resolutions[k]
            circle_resolution = int(np.ceil(
                angular_resolution * np.max(radii[:2])
            ))
            tz = np.linspace(-radii[2], radii[2], angular_resolution)
            for z in tz:
                for circle_step in range(circle_resolution):
                    theta = 2*np.pi*circle_step/circle_resolution
                    Q.append(np.array([
                        radii[0]*np.cos(theta),
                        radii[1]*np.sin(theta),
                        z
                    ]))
        Q = np.array(Q)
        # Validate
        if Q.shape[0] != self.compute_num_kernel_points():
            raise DeepLearningException(
                'KernelPointStructureInitializer generated an unexpected '
                'number of kernel points for concentric cylinders.'
            )
        # Return
        return Q

    def sample_cone(self):
        r"""
        In the cone strategy angular resolution
        :math:`r_a` is used directly to sample along the :math:`z`-axis.

        The cone will be generated as parametric circles which angular
        resolution and radius increases as they get far from the zero.


        More concretely, the cone will have a number of points as shown below:

        .. math::

            1 + \sum_{i=1}^{r_a}{\left\lceil
                r_a  \dfrac{\lvert{z_i}\rvert}{z^*}
            \right\rceil}

        Where :math:`z^*` is the maximum height for the cone.

        :return: Points sampled from concentric cones.
        :rtype: :class:`tf.Tensor`
        """
        # Generate kernel's structure
        dimensions = 3
        Q = []
        for k in range(self.radii_resolution):
            if k == 0:  # Center point
                Q.append(np.zeros(3))
                continue
            # Concentric cones
            radii = k/(self.radii_resolution-1)*self.max_radii
            angular_resolution = self.angular_resolutions[k]
            tz = np.linspace(-radii[2], radii[2], angular_resolution)
            z_ratio = np.abs(tz)/np.max(tz)
            for j, z in enumerate(tz):
                circle_resolution = int(
                    np.ceil(angular_resolution * z_ratio[j])
                )
                for circle_step in range(circle_resolution):
                    theta = 2*np.pi*circle_step/circle_resolution
                    Q.append(np.array([
                        z_ratio[j]*radii[0]*np.cos(theta),
                        z_ratio[j]*radii[1]*np.sin(theta),
                        z
                    ]))
        Q = np.array(Q)
        # Validate
        if Q.shape[0] != self.compute_num_kernel_points():
            raise DeepLearningException(
                'KernelPointStructureInitializer generated an unexpected '
                'number of kernel points for concentric cones.'
            )
        # Return
        return Q

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def compute_num_kernel_points(self):
        """
        Compute the expected number of support points for the kernel
        considering the attributes of the initializer.

        :return: Expected number of support points.
        :rtype: int
        """
        init_type = self.initialization_type.lower()
        if init_type == 'concentric_ellipsoids' or init_type == 'zeros':
            return int(np.sum(np.power(self.angular_resolutions, 2)))
        elif init_type == 'concentric_rectangulars':
            return int(1+np.sum(  # 1+ first ang.res. forced to 1 (center pt.)
                6 * np.power(self.angular_resolutions[1:], 2)
                - 12 * np.array(self.angular_resolutions[1:])
                + 8
            ))
        elif init_type == 'concentric_grids':
            return int(np.sum(np.power(self.angular_resolutions, 3)))
        elif init_type == 'concentric_cylinders':
            return int(1 + np.sum([
                self.angular_resolutions[k]*(
                    int(np.ceil(
                        self.angular_resolutions[k] *
                        np.max(
                            (
                                k/(self.radii_resolution-1)*self.max_radii
                            )[:2]
                        )
                    ))
                )
                for k in range(1, self.radii_resolution)
            ]))
        elif init_type == 'cone':
            z = np.linspace(
                -self.max_radii[2],
                self.max_radii[2],
                self.angular_resolutions[1]
            )
            return int(1 + np.sum([
                np.ceil(
                    self.angular_resolutions[1] *
                    np.abs(z[i])/self.max_radii[2]
                )
                for i in range(self.angular_resolutions[1])
            ]))
        raise DeepLearningException(
            'KernelPointStructureInitializer could not compute the number '
            'of kernel points for a kernel structure of type '
            f'"{self.initialization_type}"'
        )
