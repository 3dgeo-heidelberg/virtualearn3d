# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field import ReceptiveField
from scipy.spatial import KDTree as KDT
import numpy as np
import open3d


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldFPS(ReceptiveField):
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing a receptive field based on furthest point subsampling.

    See :class:`.ReceptiveField`, :class:`.ReceptiveFieldGS`, and
    :class:`.FurthestPointSubsamplingPreProcessor`.

    :ivar num_points: The number of points so each point cloud is subsampled
        to this number of points through FPS. Typically noted as :math:`R`.
    :vartype num_points: int
    :ivar num_encoding_neighbors: How many neighbors consider when propagating
        and reducing. Assume the number of encoding neighbors is
        :math:`m^* \in \mathbb{Z}_{\geq 0}`. For then, when reducing values
        from :math:`\pmb{X} \in \mathbb{R}^{m \times n}` (input point cloud)
        to :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` (receptive field
        points), each reduced value in :math:`\pmb{Y}` will be obtained
        by reducing :math:`m^*` values in :math:`\pmb{X}`. Also, when
        propagating values from :math:`\pmb{Y} \in \mathbb{R \times n}` to
        :math:`\pmb{X} \in \mathbb{m \times n}`, each propagated value in
        :math:`\pmb{X}` will be obtained by reducing :math:`m^*` values from
        :math:`\pmb{Y}`.
    :vartype num_encoding_neighbors: int
    :ivar fast: Flag to control whether to use the fast mode or not. When
        running the FPS receptive field in fast mode, a random uniform sampling
        is computed before the furthest poit subsampling. While faster because
        it reduces the computational burden for the FPS, this approach is also
        less stable and might produce unexpected results.
    :vartype fast: bool
    :ivar N: The indexing matrix
        :math:`\pmb{N} \in \mathbb{Z}_{\geq 0}^{R \times m^*}`. Each row
        :math:`i` in this
        matrix represents the indices in :math:`\pmb{X}` that are associated
        to the point represented by the row :math:`i` in :math:`\pmb{Y}`.
    :vartype N: :class:`np.ndarray`
    :ivar M: The reverse indexing matrix
        :math:`\pmb{M} \in \mathbb{Z}_{\geq 0}^{m \times m^*}. Each row
        :math:`i` in this matrix represents the indices in :math:`\pmb{Y}` that
        are associated to the points represented by the row :math:`i` in
        :math:`\pmb{X}`.
    :vartype M: :class:`np.ndarray`
    :ivar x: The center point of the receptive field. It is assigned when
        calling :meth:`receptive_field_fps.ReceptiveFieldFPS.fit`.
    :vartype x: :class:`np.ndarray`
    :ivar Y: The subsample representing the original input point cloud, i.e.,
        a matrix of coordinates in a :math:`n`-dimensional space such that
        :math:`\pmb{Y} \in \mathbb{R}^{R \times n}`.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize/instantiate a receptive field object.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveFieldFPS.

        :Keyword Arguments:
            *   *num_points* (``int``) --
                The number of points :math:`R` the input points must be reduced
                too.
                In other words, for a given number of input points :math:`m_1`,
                the reduced number of points will be :math:`R`. For another,
                let us say different (i.e., :math:`m_1 \neq m_2`) number of
                points, the reduced number of points will also be
                :math:`R`.
            * *num_encoding_neighbors* (``int``) --
                How many neighbors consider when doing propagations and
                reductions. For instance, for three encoding neighbors
                propagating a value means three points in the receptive
                field will be considered to estimate the value in the
                original domain. Analogously, reducing a value means three
                points in the original domain will be considered to encode
                the value in the receptive field.
            * *fast* (``bool``) --
                A flag to enable the fast-computation mode. When True, a random
                uniform subsampling will be computed before the furthest point
                sampling so the latest is faster because it is not considering
                the entire input point cloud.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.num_points = kwargs.get('num_points', 8000)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 3)
        self.fast = kwargs.get('fast', False)
        self.N = None  # The indexing matrix will be created during fit
        self.M = None  # The reverse indexing matrix will be created during fit
        self.x = None  # The center point of the receptive field
        self.Y = None  # The centroids of the receptive field

    # ---   RECEPTIVE FIELD METHODS   --- #
    # ----------------------------------- #
    def fit(self, X, x):
        """
        Fit the receptive field to represent the given points by taking the
        subset of the furthest points, i.e., the subset of points that maximize
        the distances between points. Typically, the next point in a FPS
        iteration maximizes the distance with respect to the already
        considered points in a greedy scheme.

        :param X: The input matrix of m points in an n-dimensional space.
        :type X: :class:`np.ndarray`
        :param x: The center point used to define the origin of the receptive
            field.
        :type x: :class:`np.ndarray`
        :return: The fit receptive field itself (for fluent programming)
        :rtype: :class:`.ReceptiveFieldFPS`
        """
        # Validate input
        if x is None:
            raise ValueError(
                'ReceptiveFieldFPS cannot fit without an input center point x.'
            )
        if X is None:
            raise ValueError(
                'ReceptiveFieldFPS cannot fit without input points X.'
            )
        # Center and scale the input point cloud
        self.x = x
        X = self.center_and_scale(X)
        # Compute the FPS "centroids"
        self.Y = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
            X,
            fast=self.fast,
            num_points=self.num_points
        )
        # Find the indexing matrix N
        kdt = KDT(X)
        self.N = kdt.query(self.Y, k=self.num_encoding_neighbors)[1]
        # Find the indexing matrix M
        kdt = KDT(self.Y)
        self.M = kdt.query(X, k=self.num_encoding_neighbors)[1]
        if len(self.M.shape) < 2:
            self.M = self.M.reshape(-1, 1)
        # Return self for fluent programming
        return self

    def centroids_from_points(self, X):
        """
        The centroids of an FPS receptive field are said to be the subsampled
        points themselves.

        :param X: The matrix of input points (can be NONE, in fact, it is not
            used).
        :type X: :class:`np.ndarray` or None
        :return: A matrix which rows are the points representing the centroids.
        :rtype: :class:`np.ndarray`
        """
        return self.Y

    def propagate_values(self, v, reduce_strategy='mean', **kwargs):
        r"""
        Propagate :math:`R` values associated to
        :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` to :math:`m`
        values associated to :math:`\pmb{X} \in \mathbb{R}^{m \times n}`
        through the indexing matrix
        :math:`\pmb{M} \in \mathbb{Z}_{\geq 0}^{m \times m^*}`.

        :param v: The :math:`R` values to be propagated.
        :type v: list
        :param reduce_strategy: The reduction strategy, either "mean" or
            "closest".
        :type reduce_strategy: str
        :return: The output as a matrix when there are more than two values per
            point or the output as a vector when there is one value per point.
        :rtype: :class:`np.ndarray`
        """
        # Determine the dimensionality of each value (both scalar and vectors
        # can be propagated). All values must have the same dimensionality.
        try:
            val_dim = len(v[0])
        except Exception as ex:
            val_dim = 1
        # Prepare output matrix
        Ytype = v.dtype if isinstance(v, np.ndarray) else type(v[0])
        Y = np.full([len(self.M), val_dim], 0, dtype=Ytype)
        # Populate output matrix : Reduce by mean
        if reduce_strategy == 'mean':
            for i, Mi in enumerate(self.M):
                Y[i] = np.mean(v[Mi], axis=0)
        # Populate output matrix : Take from closest
        elif reduce_strategy == 'closest':
            for i, Mi in enumerate(self.M):
                Y[i] = v[Mi[0]]
        else:  # Unexpected reduce strategy
            raise ValueError(
                'The FPS receptive field received an unexpected '
                'reduce_strategy when propagating values.'
            )
        # Return output matrix (or vector if single-column)
        return Y if Y.shape[1] > 1 else Y.flatten()

    def reduce_values(self, X, v, reduce_f=np.mean):
        r"""
        Reduce :math:`m` values associated to
        :math:`\pmb{X} \in \mathbb{R}^{m \times n} to :math:`R` values
        associated to :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` through the
        indexing matrix
        :math:`\pmb{N} \in \mathbb{Z}_{\geq 0}^{m \times m^*}`.

        :param X: The centroids representing the furthest point subsampling
            computed by the receptive field. It can be None since it is not
            used for an FPS receptive field.
        :type X: :class:`np.ndarray` or None
        :param v: The vector of values to reduce. The :math:`m` input
            components will be reduced to :math:`R` output components.
        :param reduce_f: The function to reduce many values to a single
            one. By default, it is mean.
        :type reduce_f: callable
        :return: The reduced vector.
        :rtype: :class:`np.ndarray`
        """
        # Reduce
        v_reduced = np.zeros(len(self.N))
        for i, Ni in enumerate(self.N):
            v_reduced[i] = reduce_f(v[Ni])
        # Return
        return v_reduced

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def center_and_scale(self, X):
        """
        Like :meth:`receptive_field_gs.ReceptiveFieldGS.center_and_scale` but
        without scaling, i.e., only centering.
        """
        return X - self.x

    def undo_center_and_scale(self, X):
        """
        Like :meth:`receptive_field_gs.ReceptiveFieldGS.undo_center_and_scale`
        but without scaling, i.e., only centering.
        """
        return X + self.x

    @staticmethod
    def compute_fps_on_3D_pcloud(X, num_points=None, fast=False):
        r"""
        Compute the furthest point sampling (FPS) algorithm on the point cloud
        represented by the input 3D matrix
        :math:`\pmb{X} \in \mathbb{R}^{m \times 3}`. The result is an output
        matrix :math:`\pmb{Y} \in \mathbb{R}^{R \times 3}`.

        :param X: The input 3D matrix (rows are points, columns dimensions).
        :type X: :class:`np.ndarray`
        :param num_points: The number of points :math:`R` selected through the
            furthest point sampling method.
        :type num_points: int
        :param fast: Whether to use a fast approximation of FPS (True) or
            the exact computation (False). The fast approximation is computed
            through uniform down sample.
        :type fast: bool
        :return: The subsampled point cloud.
        :rtype: :class:`np.ndarray`
        """
        o3d_cloud = open3d.geometry.PointCloud()
        o3d_cloud.points = open3d.utility.Vector3dVector(X)
        if fast:
            step = X.shape[0] // num_points
            o3d_cloud = o3d_cloud.uniform_down_sample(step)
        o3d_cloud = o3d_cloud.farthest_point_down_sample(num_points)
        if len(o3d_cloud.points) != num_points:
            raise ValueError(
                f'ReceptiveFieldFPS failed to sample {num_points}. Only '
                f'{len(o3d_cloud.points)} samples were taken for a given '
                f'input of {X.shape[0]} points.' +(
                    '' if len(o3d_cloud.points) >= X.shape[0] else
                    '\nFarthest point down sample might discard points under '
                    'some circumstances, e.g., repeated points.'
                )
            )
        return np.asarray(o3d_cloud.points)

