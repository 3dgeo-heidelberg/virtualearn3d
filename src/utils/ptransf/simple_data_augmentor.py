# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.data_augmentor import DataAugmentor
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SimpleDataAugmentor(DataAugmentor):
    """
    :author: Alberto M. Esmoris Pena

    Class representing a simple data augmentation object. It supports random
    rotations, scaling, and jitter. Each data augmentation transformation
    can be defined with a random distribution that can be either normal
    or uniform.

    :ivar transformations: List of transformations to be applied to augment
        the input data.
    :vartype transformations: list of dict
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the simple data augmentor.

        :param kwargs: The key-word specification for the initialization of the
            simple data augmentor.
        """
        self.transformations = kwargs.get('transformations', None)
        if self.transformations is None or len(self.transformations) < 1:
            raise ValueError(
                'SimpleDataAugmentor cannot be initialized without '
                'transformations.'
            )

    # ---  DATA AUGMENTATION METHODS  --- #
    # ----------------------------------- #
    def augment(self, X, **kwargs):
        r"""
        See :class:`.DataAugmentor` and :meth:`.DataAugmentor.augment`.

        :param X: It can be a single batch, e.g., a tensor
            :math:`K x m x n` that represents :math:`K` point clouds of
            :math:`m` points in a :math:`n`-dimensional space. Alternatively,
            it can be a list of batch elements, such that all the elements
            in the list must be consistently augmented. For example, assume
            [X1, X2, X3] is given where
            :math:`\mathcal{X}_i \in \mathbb{R}^{K_i \times m_i \times n}` for
            :math:`i=1,\ldots,3`. Then the matrices :math:`\pmb{X}_{1j}`,
            :math:`\pmb{X}_{2j}`, and :math:`\pmb{X}_{3j}` must be augmented
            in the same way, i.e., using the same samples from random
            distributions, for rotations and scaling. However, jitter will
            be applied independently to any element.
        :type X: list or :class:`np.ndarray`
        """
        for transformation in self.transformations:
            X = SimpleDataAugmentor.apply_transformation(transformation, X)
        return X

    # ---  TRANSFORMATION METHODS  --- #
    # -------------------------------- #
    @staticmethod
    def apply_transformation(transformation, X):
        """
        Apply the given transformation to the structure space matrices.

        NOTE that the transformation is applied in place, despite the
        transformed space is also returned.

        :param transformation: The specification of the transformation to be
            applied.
        :type transformation: dict
        :param X: The structure spaces to be transformed.
        :type X: list or :class:`np.ndarray`
        :return: The transformed structure spaces.
        :rtype: list or :class:`np.ndarray`
        """
        ttype = transformation['type']
        ttype_low = ttype.lower()
        if ttype_low == 'rotation':
            return SimpleDataAugmentor.apply_rotation(transformation, X)
        elif ttype_low == 'scale':
            return SimpleDataAugmentor.apply_scale(transformation, X)
        elif ttype_low == 'jitter':
            return SimpleDataAugmentor.apply_jitter(transformation, X)
        else:
            raise ValueError(
                'SimpleDataAugmentor received an unexpected transformation '
                f'type: "{ttype}"'
            )

    @staticmethod
    def apply_rotation(transformation, X):
        """
        Apply a rotation transformation.

        :param transformation: The rotation specification.
        :type transformation: dict
        :param X: The structure spaces to be transformed.
        :type X: list or :class:`np.ndarray`
        :return: The rotated structure spaces.
        :rtype: list or :class:`np.ndarray`
        """
        # Determine number of inputs, space dimensionality, and rotation axis
        m = X[0].shape[0] if isinstance(X, list) else X.shape[0]
        nx = X[0].shape[-1] if isinstance(X, list) else X.shape[-1]
        if nx != 3:
            raise ValueError(
                'SimpleDataAugmentor for structure spaces with a '
                'dimensionality other than 3D does not support rotations.'
            )
        axis = np.array(transformation.get('axis', [0, 0, 1]))
        # The random rotation angle for each structure space in the input
        angles = SimpleDataAugmentor.compute_distribution(
            transformation['angle_distribution'],
            dim=m
        )
        # Apply the corresponding rotation to each structure space
        R = np.zeros((nx, nx))
        if isinstance(X, list):
            for i in range(m):
                SimpleDataAugmentor.set_rotation_matrix(R, axis, angles[i])
                for k in range(len(X)):
                    X[k][i] = (R@X[k][i].T).T
        else:
            for i, Xi in X:
                SimpleDataAugmentor.set_rotation_matrix(R, axis, angles[i])
                X[i] = (R@Xi.T).T
        # Return rotated structure spaces
        return X

    @staticmethod
    def apply_scale(transformation, X):
        """
        Apply a scaling transformation.

        :param transformation: The scaling specification.
        :param X: The structure spaces to be transformed.
        :type X: list or :class:`np.ndarray`
        :return: The scaled structure spaces.
        :rtype: list or :class:`np.ndarray`
        """
        # Compute random scale factors
        m = X[0].shape[0] if isinstance(X, list) else X.shape[0]  # Num. inputs
        scale_factors = SimpleDataAugmentor.compute_distribution(
            transformation['scale_distribution'],
            dim=m
        )
        # Apply the corresponding random scaling to each structure space
        if isinstance(X, list):
            for i in range(m):
                scale_factor = scale_factors[i]
                for k in range(len(X)):
                    X[k][i] = scale_factor*X[k][i]
        else:
            for i, Xi in X:
                X[i] = scale_factors[i]*Xi
        # Return scaled structure spaces
        return X

    @staticmethod
    def apply_jitter(transformation, X):
        """
        Apply a jitter transformation.

        :param transformation: The jitter specification.
        :param X: The structure spaces to be transformed.
        :type X: list or :class:`np.ndarray`
        :return: The structure spaces with jitter.
        :rtype: list or :class:`np.ndarray`
        """
        # Determine number of inputs
        m = X[0].shape[0] if isinstance(X, list) else X.shape[0]
        # Apply the corresponding jitter to each structure space
        if isinstance(X, list):
            for i in range(m):
                for k in range(len(X)):
                    # Compute jitter distribution for current structure space
                    Jki = SimpleDataAugmentor.compute_distribution(
                        transformation['noise_distribution'],
                        dim=X[k][i].shape
                    )
                    # Add jitter to the current structure space
                    X[k][i] += Jki
        else:
            for i, Xi in X:
                # Compute jitter distribution for current structure space
                Ji = SimpleDataAugmentor.compute_distribution(
                    transformation['noise_distribution'],
                    dim=X[i].shape
                )
                # Add jitter to the current structure space
                X[i] += Ji
        # Return structure space with jitter
        return X

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def compute_distribution(distribution, dim=1):
        """
        Compute the distribution corresponding to the given specification.

        :param distribution: The specification of the distribution to be
            computed.
        :type distribution: dict
        :param dim: The sampling dimensionality, i.e., how many samples will
            be taken from the distribution.
        :return: A tensor of samples representing the computed
            distribution.
        :rtype: :class:`np.ndarray`
        """
        # Get distribution type
        type = distribution['type']
        type_low = type.lower()
        # Compute uniform distribution
        if type_low == 'uniform':
            return np.random.uniform(
                distribution['start'],
                distribution['end'],
                dim
            )
        # Compute normal distribution
        elif type_low == 'normal' or type_low == 'gaussian':
            return np.random.normal(
                distribution['mean'],
                distribution['stdev'],
                dim
            )
        # Handle unexpected distribution
        else:
            raise ValueError(
                'SimpleDataAugmentor.compute_distribution received an '
                'unexpected distribution: "{type}"'
            )

    @staticmethod
    def set_rotation_matrix(R, axis, angle):
        """
        Update the rotation matrix in place, so it represents a rotation
            around the given axis with the given angle.

        :param R: The rotation matrix to be updated.
        :type R: :class:`np.ndarray`
        :param axis: The rotation axis.
        :type axis: list or tuple or :class:`np.ndarray`
        :param angle: The angle of the rotation.
        :typa angle: float
        :return: The updated rotation matrix (it is also updated in place).
        :rtype: :class:`np.ndarray`
        """
        # Precomputations
        cos, sin = np.cos(angle), np.sin(angle)
        cos_com = 1-cos
        # Set the 3D rotation matrix around an arbitrary axis
        R[0, 0] = cos + axis[0]*axis[0]*cos_com
        R[0, 1] = axis[0]*axis[1]*cos_com - axis[2]*sin
        R[0, 2] = axis[0]*axis[2]*cos_com + axis[1]*sin
        R[1, 0] = axis[0]*axis[1]*cos_com + axis[2]*sin
        R[1, 1] = cos + axis[1]*axis[1]*cos_com
        R[1, 2] = axis[1]*axis[2]*cos_com - axis[0]*sin
        R[2, 0] = axis[0]*axis[2]*cos_com - axis[1]*sin
        R[2, 1] = axis[1]*axis[2]*cos_com + axis[0]*sin
        R[2, 2] = cos + axis[2]*axis[2]*cos_com
        # Return updated rotation matrix
        return R
