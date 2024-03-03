# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.initializer.initializer import Initializer
import tensorflow as tf
import scipy.optimize as opt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class KernelPointBallInitializer(Initializer):
    r"""
    :author: Alberto M. Esmoris Pena

    A kernel point ball initializer initializes the structure space matrix
    representing the support points of a kernel. It will generate a
    structure space matrix :math:`\pmb{Q} \in \mathbb{R}^{m_q \times 3}`.
    This matrix represents :math:`m_q` points in a 3D Euclidean space inside a
    ball of radius :math:`r` such that the distances between points is
    maximized.

    The points are distributed inside a ball solving an energy minimization
    problem for an energy model:

    .. math::
        E = \sum_{i=1}^{m_q}{\biggl(
            \lVert\pmb{q}_{i*}\rVert^2 + \sum_{j=1,j\neq{i}}^{m_q}{
                \lVert{
                    \pmb{q}_{i*} - \pmb{q}_{j*}
                }\rVert^{-1}
            }
        \biggr)}

    Note for a 3D space, the gradient of this energy model is defined as:

    .. math::
        \nabla_{\pmb{Q}}E = \left[\begin{array}{ccc}
            \dfrac{\partial E}{\partial q_{1x}} &
                \dfrac{\partial E}{\partial q_{1y}} &
                \dfrac{\partial E}{\partial q_{1z}} \\
            \vdots & \vdots & \vdots \\
            \dfrac{\partial E}{\partial q_{m_q x}} &
                \dfrac{\partial E}{\partial q_{m_q y}} &
                \dfrac{\partial E}{\partial q_{m_q z}} &
        \end{array}\right]

    For then, any partial derivative in the gradient can be computed as:

    .. math::
        \dfrac{\partial E}{\partial q_{ik}} = 2 \left[
            q_{ik} + \sum_{j=1,j\neq{i}}^{m_q}{\dfrac{
                q_{jk} - q_{ik}
            }{
                \lVert{\pmb{q}_{i*} - \pmb{q}_{j*}}\rVert^3
            }}
        \right]

    Intuitively, this model can be thought as if every point caused a repulsive
    force on the others while the zero point imposes an attractive force on
    all the points. The minimization of the energy through a conjugate gradient
    method leads to the desired disposition of the points in :math:`\pmb{Q}`.

    :ivar target_radius: The radius of the ball to which the kernel points
        belong (:math:`r`).
    :vartype target_radius: float
    :ivar num_points: The number of points representing the kernel
        :math:`m_q`.
    :vartype num_points: int
    :ivar deformable: Whether to allow the neural network to update the
        structure space of the kernel (i.e., the coordinates of the kernel
        points) or not.
    :vartype deformable: bool
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        target_radius=1.0,
        num_points=19,
        deformable=False,
        **kwargs
    ):
        """
        Initialize the member attributes of the initializer.

        :param target_radius: The radius of the kernel's ball.
        :type target_radius: float
        :param num_points: The number of kernel points.
        :type num_points: int
        :param deformable: True to make the kernel points trainable by the
            neural network, False otherwise.
        :type deformable: bool
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.target_radius = target_radius
        self.num_points = num_points
        self.deformable = deformable
        self.name = kwargs.get('name', 'Q')

    # ---   INITIALIZER METHODS   --- #
    # ------------------------------- #
    def __call__(self, shape=None, dtype='float32'):
        r"""
        Initialize a ball-like structure space matrix representing the support
        points of a kernel.

        :param shape: The shape parameter is ignored as the dimensionality of
            the initialized matrix depends on the arguments used to build
            the initializer.
        :return: The structure space matrix for a ball-like kernel point
            :math:`\pmb{Q} \in \mathbb{R}^{m_q \times 3}`.
        :rtype: :class:`tf.Tensor`
        """
        # Random initialization
        init_radius = 1.0  # Radius for random initialization
        Q = np.random.rand(self.num_points-1, 3) * 2*init_radius - init_radius
        x = Q.flatten()  # all points as a single vector
        # Minimize through conjugate gradient
        x = opt.minimize(
            KernelPointBallInitializer.energy_f,
            x,
            method='CG',
            jac=KernelPointBallInitializer.energy_df
        ).x
        Q = x.reshape((-1, 3))
        # To target radius
        current_radius = np.sqrt(np.max(np.sum(np.square(Q), axis=1)))
        Q = Q * self.target_radius / current_radius
        # First point is zero
        Q = np.vstack([np.zeros((1, 3)), Q])
        # Return initialized kernel structure as tensor
        return tf.Variable(
            Q,
            dtype=dtype,
            trainable=self.deformable,
            name=self.name
        )

    # ---   ENERGY MODEL   --- #
    # ------------------------ #
    @staticmethod
    def energy_f(x):
        """
        Compute the energy model.

        :param x: The vectorized matrix of input points.
        :type x: :class:`np.ndarray`
        :return: The energy.
        """
        Q = np.reshape(x, (-1, 3))
        Qsq = np.square(Q)
        num_points = Q.shape[0]
        energy = 0
        for i in range(num_points):  # For each point
            energy += np.sum(Qsq[i, :])
            for j in range(num_points):  # For each point
                if i == j:  # Skip because points do not repel themselves
                    continue
                energy += 1/np.linalg.norm(Q[i]-Q[j])
        return energy

    @staticmethod
    def energy_df(x):
        """
        Compute the gradient of the energy model.

        :param x: The vectorized matrix of input points.
        :type x: :class:`np.ndarray`
        :return: The gradient of the energy.
        """
        Q = np.reshape(x, (-1, 3))
        num_points = Q.shape[0]
        DEDX = np.zeros(Q.shape)
        for i in range(num_points):  # For each point
            for k in range(3):  # For each dimension
                DEDX[i, k] += 2*Q[i, k]
                for j in range(num_points):  # For each point
                    if i == j: # Skip because points do not repel themselves
                        continue
                    DEDX[i, k] += 2*(Q[j, k]-Q[i, k])/np.power(
                        np.sum(np.square(Q[i, :]-Q[j, :])),
                        3/2
                    )
        return DEDX.flatten()
