# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from scipy.spatial import KDTree as KDT
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SupportNeighborhoods:
    r"""
    :author: Alberto M. Esmoris Pena

    Class providing the methods to generate support neighborhoods, i.e.,
    neighborhoods centered on support points (not necessarily one per point in
    the point cloud). An instance of :class:`.SupportNeighborhoods` must
    have a neighborhood definition and might potentially redefine some
    default parameters to provide alternative behaviors (e.g., the number of
    desired support points, or the number of threads to use for parallel
    computations).

    :ivar neighborhood_spec: The neighborhood specification governing the
        behavior of the support neighborhoods object. See the JSON below for
        an example:


        .. code-block:: JSON

            {
                "type": "sphere",
                "radius": 5.0,
                "separation_factor": 1.0
            }

    :vartype neighborhood_spec: dict
    :ivar support_strategy: The strategy to be used to compute the support
        points when no training class distribution has been given. It can be
        "grid" (default) to get support points through grid sampling, or "fps"
        to use furthest point sampling.
    :vartype support_strategy: str
    :ivar support_strategy_num_points: The number of points to consider when
        using a furthest point sampling strategy to compute the support points.
    :vartype support_strategy_num_points: int
    :ivar support_strategy_fast: True to use a fast approximation based on
        random sampling to compute the furthest point sampling (it works well
        provided that the number of points is big enough, e.g., > 10^3).
    :vartype support_strategy_fast: bool
    :ivar support_chunk_size: The number of support points per chunk :math:`n`.
        If zero, all support points are considered at once.
        If greater than zero, the support points will be considered in chunks
        of :math:`n` points.
    :vartype support_chunk_size: int
    :ivar training_class_distribution: The target class distribution to
        consider when building the support points. The element i of this
        vector (or list) specifies how many neighborhoods centered on a point
        of class i must be considered. If None, then the point-wise classes
        will not be considered when building the neighborhoods.
    :vartype training_class_distribution: list or tuple or :class:`np.ndarray`
    :ivar center_on_pcloud: When True, the support points defining the
        receptive field will be transformed to be a point from the input
        point cloud. Consequently, the generated neighborhoods are centered on
        points that belong to the point cloud instead of arbitrary support
        points. When False, support points do not necessarily match points
        from the point cloud.
    :vartype center_on_pcloud: bool
    :ivar nthreads: The number of threads to consider for the parallel
        computation of the support neighborhoods.
    :vartype nthreads: int
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, neighborhood_spec, **kwargs):
        """
        Initialization/instantiation of a SupportNeighborhoods object.

        See :class:`.FurthestPointSubsamplingPreProcessor`,
        :meth:`furthest_point_subsampling_pre_processor.FurthestPointSubsamplingPreProcessor.find_neighborhood`,
        :class:`.HierarchicalFPSPreProcessor`, and
        :meth:`hierarchical_fps_pre_processor.HierarchicalFpsPreProcessor.find_neighborhoood`.

        :param neighborhood_spec: The neighborhood specification governing
            the behavior of the support neighborhoods object.
        :type neighborhood_spec: dict
        :param kwargs: The key-word arguments defining the support
            neighborhoods object.
        :type kwargs: dict
        """
        self.neighborhood_spec = neighborhood_spec
        self.support_strategy = kwargs.get('support_strategy', 'grid')
        self.support_strategy_num_points = kwargs.get(
            'support_strategy_num_points', 1000
        )
        self.support_strategy_fast = kwargs.get('support_strategy_fast', False)
        self.support_chunk_size = kwargs.get('support_chunk_size', 0)
        self.training_class_distribution = kwargs.get(
            'training_class_distribution', None
        )
        self.center_on_pcloud = kwargs.get('center_on_pcloud', False)
        self.nthreads = kwargs.get('nthreads', 1)

    # ---  NEIGHBORHOOD COMPUTATION METHODS  --- #
    # ------------------------------------------ #
    def compute(self, X, y=None):
        r"""
        Compute/find the requested neighborhoods in the given input point cloud
        represented by the matrix of coordinates :math:`\pmb{X}`.

        :param X: The matrix of coordinates.
        :type X: :class:`np.ndarray`
        :param y: The vector of expected values (generally, class labels).
            It is an OPTIONAL argument that is only necessary when the
            neighborhoods must be found following a given class distribution.
        :type y: :class:`np.ndarray`
        :return: A tuple which first element are the support points
            representing the centers of the neighborhoods and which second
            element is a list of neighborhoods, where each neighborhood is
            represented by a list of indices corresponding to the rows (points)
            in :math:`\pmb{X}` that compose the neighborhood.
        :rtype: tuple
        """
        # Handle neighborhood finding
        sup_X, I = None, None
        ngbhd_type = self.neighborhood_spec['type']
        ngbhd_type_low = ngbhd_type.lower()
        class_distr = self.training_class_distribution if y is not None \
            else None
        # Note also used in HeightFeatsMiner.compute_height_features_on_support
        if self.neighborhood_spec['radius'] == 0:
            # The neighborhood of radius 0 is said to be the entire point cloud
            I = [np.arange(len(X), dtype=int).tolist()]
            sup_X = np.mean(X, axis=0).reshape(1, -1)
        elif ngbhd_type_low == 'cylinder':
            # Circumference boundary on xy, infinite on z
            X2D = X[:, :2]
            sup_X = GridSubsamplingPreProcessor.build_support_points(
                X2D,
                self.neighborhood_spec['separation_factor'],
                self.neighborhood_spec['radius'],
                support_strategy=self.support_strategy,
                support_strategy_num_points=self.support_strategy_num_points,
                support_strategy_fast=self.support_strategy_fast,
                y=y,
                class_distr=class_distr,
                center_on_X=self.center_on_pcloud,
                nthreads=self.nthreads
            )
            kdt = KDT(X2D)
            kdt_sup = KDT(sup_X)
            I = kdt_sup.query_ball_tree(kdt, self.neighborhood_spec['radius'])
            sup_X = np.hstack([sup_X, np.zeros((sup_X.shape[0], 1))])
        elif ngbhd_type_low == 'sphere':
            # Spheres with a greater than zero radius
            sup_X = GridSubsamplingPreProcessor.build_support_points(
                X,
                self.neighborhood_spec['separation_factor'],
                self.neighborhood_spec['radius'],
                support_strategy=self.support_strategy,
                support_strategy_num_points=self.support_strategy_num_points,
                y=y,
                class_distr=class_distr,
                center_on_X=self.center_on_pcloud,
                nthreads=self.nthreads
            )
            kdt = KDT(X)
            kdt_sup = KDT(sup_X)
            I = kdt_sup.query_ball_tree(kdt, self.neighborhood_spec['radius'])
        elif ngbhd_type_low == 'rectangular2d':
            # 2D rectangular boundary on xy, infinite on z
            X2D = X[:, :2]
            radius = self.neighborhood_spec['radius']
            if not isinstance(radius, list):
                radius = [radius, radius]
            sup_X = GridSubsamplingPreProcessor.build_support_points(
                X2D,
                self.neighborhood_spec['separation_factor'],
                np.min(radius),
                support_strategy=self.support_strategy,
                support_strategy_num_points=self.support_strategy_num_points,
                y=y,
                class_distr=class_distr,
                center_on_X=self.center_on_pcloud,
                nthreads=self.nthreads
            )
            # Compute the min radius cylindrical neighborhood that contains the
            # rectangular prism with infinite height
            boundary_radius = np.sqrt(
                radius[0]*radius[0]+radius[1]*radius[1]
            )
            kdt = KDT(X2D)
            kdt_sup = KDT(sup_X)
            I = kdt_sup.query_ball_tree(kdt, boundary_radius)
            # Discard points outside the 2D rectangular boundary
            XY = [X2D[Ii][:, 0:2] - sup_X[i] for i, Ii in enumerate(I)]
            mask = [
                (XYi[:, 0] >= -radius[0]) * (XYi[:, 0] <= radius[0]) *
                (XYi[:, 1] >= -radius[1]) * (XYi[:, 1] <= radius[1])
                for XYi in XY
            ]
            I = [np.array(Ii)[mask[i]].tolist() for i, Ii in enumerate(I)]
            # Fill missing 3D coordinate (z) with zero
            sup_X = np.hstack([sup_X, np.zeros((sup_X.shape[0], 1))])
        elif ngbhd_type_low == 'rectangular3d':
            # 3D rectangular boundary (voxel if all axis share the same length)
            radius = self.neighborhood_spec['radius']
            if not isinstance(radius, list):
                radius = [radius, radius, radius]
            sup_X = GridSubsamplingPreProcessor.build_support_points(
                X,
                self.neighborhood_spec['separation_factor'],
                np.min(radius),
                support_strategy=self.support_strategy,
                support_strategy_num_points=self.support_strategy_num_points,
                y=y,
                class_distr=class_distr,
                center_on_X=self.center_on_pcloud,
                nthreads=self.nthreads
            )
            # Compute the min radius spherical neighborhood that contains the
            # rectangular prism
            boundary_radius = np.sqrt(
                radius[0]*radius[0]+radius[1]*radius[1]+radius[2]*radius[2]
            )
            kdt = KDT(X)
            Iout = []
            num_chunks, chunk_size = 1,  len(sup_X)
            if self.support_chunk_size > 0:
                chunk_size = self.support_chunk_size
                num_chunks = int(np.ceil(len(sup_X)/chunk_size))
            for chunk_idx in range(num_chunks):
                # Extract chunk
                sup_idx_a = chunk_idx*chunk_size
                sup_idx_b = min(
                    (chunk_idx+1)*chunk_size,
                    len(sup_X)
                )
                chunk_sup_X = sup_X[sup_idx_a:sup_idx_b]
                # Operate on chunk
                kdt_sup = KDT(chunk_sup_X)
                chunk_I = kdt_sup.query_ball_tree(kdt, boundary_radius)
                # Discard points outside 3D rectangular boundary
                XYZ = [X[Ii] - chunk_sup_X[i] for i, Ii in enumerate(chunk_I)]
                mask = [
                    (XYZi[:, 0] >= -radius[0]) * (XYZi[:, 0] <= radius[0]) *
                    (XYZi[:, 1] >= -radius[1]) * (XYZi[:, 1] <= radius[1]) *
                    (XYZi[:, 2] >= -radius[2]) * (XYZi[:, 2] <= radius[2])
                    for XYZi in XYZ
                ]
                chunk_I = [
                    np.array(Ii)[mask[i]].tolist()
                    for i, Ii in enumerate(chunk_I)
                ]
                Iout = Iout + chunk_I
            I = Iout
        else:
            raise ValueError(
                'SupportNeighborhoods object does not expect a '
                f'neighborhood specification of type "{ngbhd_type}"'
            )
        # Return found neighborhood
        return sup_X, I
