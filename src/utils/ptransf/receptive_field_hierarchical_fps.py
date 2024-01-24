# ---   IMPORTS   --- #
# ⨪------------------ #
from src.utils.ptransf.receptive_field import ReceptiveField
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from scipy.spatial import KDTree as KDT
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldHierarchicalFPS(ReceptiveField):
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing a hierarchical receptive field based on furthest point
    subsampling.

    A hierarchical receptive field is a special type of receptive field because
    it is composed of many receptive fields organized in a hierarchical manner.

    See :class:`.ReceptiveField`, :class:`.ReceptiveFieldFPS`, and
    :class:`.HierarchicalFPSPreProcessor`.

    :ivar num_points_per_depth: The number of points
        :math:`R_d \in \mathbb{Z}_{\geq 0}` at depth :math:`d`.
        In other words, for a given number of input points :math:`m`,
        and a hierarchy of max depth :math:`d^*` the reduced number of
        points will be :math:`m \geq R_1 \geq \ldots \geq R_{d^*}`.
        The value of :math:`R_d` will be the same, even for
        different numbers of input points.
    :vartype num_points_per_depth: list or tuple or :class:`np.ndarray` of int
    :ivar fast_flag_per_depth: A flag for each depth level to enable the
        fast-computation mode. When True, a random uniform subsampling will be
        computed before the furthest point sampling so the latest is faster
        because it is not considering the entire input point cloud.
    :vartype fast_flag_per_depth: list or tuple or :class:`np.ndarray` of bool
    :ivar num_downsampling_neighbors: How many neighbors consider at each depth
        to compute the subsampling. For each depth, the number specifies how
        many points from the source space will be involved in the
        computation of each point in the downsampled space. The
        neighborhoods are made of the k-nearest neighbors. Note that
        the first value in this list corresponds to the
        ``num_encoding_neighbors`` attribute of the
        :class:`.ReceptiveFieldFPS` receptive field.
    :vartype num_downsampling_neighbors: list or tuple or :class:`np.ndarray`
        of int
    :ivar num_pwise_neighbors: How many nearest neighbors consider at each
        depth-level. In other words, for each point in the structure space at
        any given depth, the number of nearest neighbors in the same
        structure space that must be considered.
    :vartype num_pwise_neighbors: list or tuple or :class:`np.ndarray` of int
    :ivar num_upsampling_neighbors: How many neighbors consider at each depth
        to compute the upsampling. For each depth, the number specifies how
        many points from the source space will be involved in the
        computation of each point in the upsampled space. The
        neighborhoods are made of the k-nearest neighbors. Note that
        the first value in this list corresponds to the reverse
        indexing matrix (``M`` attribute) of the
        :class:`.ReceptiveFieldFPS` receptive field.
    :vartype num_upsampling_neighbors: list or tuple or :class:`np.ndarray`
        of int
    :ivar max_depth: The max depth of the hierarchy, i.e., how many receptive
        fields.
    :vartype max_depth: int
    :ivar NDs: The :math:`\pmb{N}^D_d` matrices of
        indices for downsampling with depth :math:`d=1,\ldots,d^*`.
        More concretely, :math:`n^D_{dij}`
        is the index of the j-th neighbor in the structure space before
        the downsampling of the i-th point in the downsampled structure space
        (at depth :math:`d`).
        It can be seen as an indexing tensor whose slices can be seen as
        indexing matrices.
    :vartype NDs: list of :class:`np.ndarray` of int
    :ivar Ns: The :math:`\pmb{N}_d` matrices of indices for neighborhoods
        with depth :math:`d=1,\ldots,d^*`.
        More concretely, :math:`n_{dij}` is the index of the j-th neighbor of
        the i-th point in the structure space at depth :math:`d`.
        It can be seen as an indexing tensor whose slices can be seen as
        indexing matrices.
    :vartype Ns: list of :class:`np.ndarray` of int
    :ivar NUs: The :math:`\pmb{N}^U_d` matrices of indices for upsampling with
        depth :math:`d=1,\ldots,d^*`.
        More concretely, :math:`n^U_{dij}` is the index of the j-th neighbor
        in the structure space before the upsampling of the i-th point in the
        upsampled structure space (at depth :math:`d`).
    :vartype NUs: list of :class:`np.ndarray` of int
    :ivar x: The center point of the receptive field. It is assigned when
        calling :meth:`receptive_field_fps.ReceptiveFieldFPS.fit`.
    :vartype x: :class:`np.ndarray`
    :ivar Ys: The hierarchical subsamples representing the original input point
        cloud and, i.e., a matrices of coordinates in a :math:`n_x`-dimensional
        space such that :math:`\pmb{Y}_d \in \mathbb{R}^{R_d \times n_x}`.
    :vartype Ys: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize/instantiate a hierarchical receptive field object.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveFieldHierarchicalFPS.

        :Keyword Arguments:
            *   *num_points_per_depth* (``list or tuple or np.ndarray of int``) --
                The number of points :math:`R_d \in \mathbb{Z}_{\geq 0}` at
                depth :math:`d`.
                In other words, for a given number of input points :math:`m`,
                and a hierarchy of max depth :math:`d^*` the reduced number of
                points will be :math:`m \geq R_1 \geq \ldots \geq R_{d^*}`.
                The value of :math:`R_d` will be the same, even for
                different numbers of input points.
            *   *fast_flag_per_depth* (``list or tuple or np.ndarray of
                bool``) --
                A flag for each depth level to enable the fast-computation
                mode. When True, a random uniform subsampling will be computed
                before the furthest point sampling so the latest is faster
                because it is not considering the entire input point cloud.
            *   *num_downsampling_neighbors* (``list or tuple or np.ndarray of
                int``) --
                How many neighbors consider at each depth to compute the
                subsampling. For each depth, the number specifies how many
                points from the source space will be involved in the
                computation of each point in the downsampled space. The
                neighborhoods are made of the k-nearest neighbors. Note that
                the first value in this list corresponds to the
                ``num_encoding_neighbors`` attribute of the
                :class:`.ReceptiveFieldFPS` receptive field.
            *   *num_pwise_neighbors* (``list or tuple or np.ndarray of
                int``) --
                How many nearest neighbors consider at each depth-level. In
                other words, for each point in the structure space at any
                given depth, the number of nearest neighbors in the same
                structure space that must be considered.
            *   *num_upsampling_neighbors* (``list or tuple or np.ndarray of
                int``) --
                How many neighbors consider at each depth to compute the
                upsampling. For each depth, the number specifies how many
                points from the source space will be involved in the
                computation of each point in the upsampled space. The
                neighborhoods are made of the k-nearest neighbors. Note that
                the first value in this list corresponds to the reverse
                indexing matrix (``M`` attribute) of the
                :class:`.ReceptiveFieldFPS` receptive field.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.num_points_per_depth = kwargs.get(
            'num_points_per_depth', [1024, 512, 256, 128, 64]
        )
        self.fast_flag_per_depth = kwargs.get(
            'fast_flag_per_depth', [False]*5
        )
        self.num_downsampling_neighbors = kwargs.get(
            'num_downsampling_neighbors', [1, 16, 8, 8, 4]
        )
        self.num_pwise_neighbors = kwargs.get(
            'num_pwise_neighbors', [64, 32, 32, 16, 8]
        )
        self.num_upsampling_neighbors = kwargs.get(
            'num_upsampling_neighbors', [1, 16, 8, 8, 4]
        )
        self.max_depth = len(self.num_points_per_depth)
        self.NDs = [  # The downsampling matrices of indices (computed at fit)
            None for i in range(self.max_depth)
        ]
        self.Ns = [  # The pwise neighborhood matrices of indices (comp at fit)
            None for i in range(self.max_depth)
        ]
        self.NUs = [  # The upsampling matrices of indices (computed at fit)
            None for i in range(self.max_depth)
        ]
        self.x = None  # The center point of the receptive field
        self.Ys = None  # The centroids of the hierarchical receptive fields

    # ---  RECEPTIVE FIELD METHODS  --- #
    # --------------------------------- #
    def fit(self, X, x):
        r"""
        Fit the receptive field to represent the given points by taking the
        subset of the furthest points in a recursive way leading to a hierarchy
        of receptive fields.

        See :class:`.ReceptiveFieldFPS` and
        :meth:`receptive_field_fps.ReceptiveFieldFPS.fit`.

        :param X: The input matrix of :math:`m` points in a
            :math:`n_x`-dimensional space (for now, :math:`n_x=3`).
        :type X: :class:`np.ndarray`
        :param x: The center point used to define the origin of the
            hierarchical receptive field.
        :type x: :class:`np.ndarray`
        :return: The fitted receptive field itself (for fluent programming).
        :rtype: :class:`.ReceptiveFieldHierarchicalFPS`
        """
        # Validate input
        if x is None:
            raise ValueError(
                'ReceptiveFieldHierarchicalFPS cannot fit without an input '
                'center point x.'
            )
        if X is None:
            raise ValueError(
                'ReceptiveFieldHierarchicalFPS cannot fit without input '
                'points X.'
            )
        # Center and scale the input point cloud
        self.x = x
        X = self.center_and_scale(X)
        # Recursively compute the many FPS receptive fields
        Xd = X
        for d in range(self.max_depth):
            # Compute the FPS "centroids" at depth d (Yd)
            self.Ys[d] = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
                Xd,
                fast=self.fast_flag_per_depth[d],
                num_points=self.num_points_per_depth[d]
            )
            # Find the downsampling matrix at depth d (NDd)
            kdt = KDT(Xd)
            self.NDs[d] = kdt.query(
                self.Ys[d], k=self.num_downsampling_neighbors[d]
            )[1]
            # TODO Rethink : Reshape NDd like NUd below ?
            # Find the upsampling matrix at depth d (NUd)
            kdt = KDT(self.Ys[d])
            NUd = kdt.query(Xd, k=self.num_upsampling_neighbors[d])[1]
            if len(NUd.shape) < 2:
                NUd = NUd.reshape(-1, 1)
            self.NUs[d] = NUd
            # Find the point-wise neighborhood matrix at depth d (Nd)
            Nd = kdt.query(self.Ys[d], k=self.num_pwise_neighbors[d])[1]
            if len(Nd.shape) < 2:
                Nd = Nd.reshape(-1, 1)
            self.Ns[d] = Nd
        # Return self for fluent programming
        return self

    def centroids_from_points(self, X):
        """
        The centroids of a hierarchical FPS receptive field are said to be
        subsampled points themselves, as for the :class:`.ReceptiveFieldFPS`.

        :param X: The matrix of input points (can be NONE, in fact, it is not
            used).
        :type X: :class:`np.ndarray` or None
        :return: A matrix which rows are the points representing the centroids.
        :rtype: :class:`np.ndarray`
        """
        return self.Ys[0]

    def propagate_values(self, v, reduce_strategy='mean', **kwargs):
        """
        See :class:`.ReceptiveFieldFPS` and
        :meth:`receptive_field_fps.ReceptiveFieldFPS.propagate_values`.
        """
        return ReceptiveFieldFPS.do_propagate_values(
            self.NUs[0], v, reduce_strategy
        )

    def reduce_values(self, X, v, reduce_f=np.mean):
        """
        See :class:`.ReceptiveFieldFPS` and
        :meth:`receptive_field_fps.ReceptiveFieldFPS.reduce_values`.
        """
        return ReceptiveFieldFPS.do_reduce_values(self.NDs[0], X, v, reduce_f)

    # ---  HIERARCHICAL RECEPTIVE FIELD METHODS  --- #
    # ---------------------------------------------- #
    def get_downsampling_matrices(self):
        r"""
        Obtain the downsampling matrices
        :math:`\pmb{N}^D_1, \ldots, \pmb{N}^D_{d^{*}}`.

        :return: The downsampling matrices.
        :rtype: list of (list or tuple or :class:`np.ndarray`) of int
        """
        return self.NDs

    def get_neighborhood_matrices(self):
        r"""
        Obtain the point-wise neighborhoods as matrices of indices
        :math:`\pmb{N}_1, \ldots, \pmb{N}_{d^{*}}`.

        :return: The matrix of indices representing the point-wise
            neighborhoods.
        :rtype: list of (list or tuple or :class:`np.ndarray`) of int
        """
        return self.Ns

    def get_upsampling_matrices(self):
        r"""
        Obtain the upsampling matrices
        :math:`\pmb{N}^U_1, \ldots, \pmb{N}^U_{d^{*}}`.

        :return: The upsampling matrices.
        :rtype: list of (list or tuple or :class:`np.ndarray`) of int
        """
        return self.NUs

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def center_and_scale(self, X):
        """
        Like :meth:`receptive_field_fps.ReceptiveFieldFPS.center_and_scale`.
        """
        return X - self.x

    def undo_center_and_scale(self, X):
        """
        Like
        :meth:`receptive_field_fps.ReceptiveFieldFPS.undo_center_and_scale`.
        """
        return X + self.x
