# ---   IMPORTS   --- #
# ------------------- #
from src.clustering.clusterer import Clusterer, ClusteringException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import numpy as np
import open3d
import time


# ---   CLASS   --- #
# ----------------- #
class DBScanClusterer(Clusterer):
    r"""
    :author: Alberto M. Esmorís Pena

    DBScan clustering on the structure space
    :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}`. It supports filtering by
    discrete categorical values (e.g., classifications), i.e., one
    DBScan on the subspace of the Euclidean space that contains only points
    belonging to a given cluster (classes, and categorical predictions are
    clusters in this context).

    More formally, let :math:`\pmb{x_{i*}} \in \mathbb{R}^{n_x}` be a point in
    the structure space, with :math:`y_i \in \mathbb{Z}_{\geq 0}` the
    integer that represents the cluster to which point :math:`i` belongs.

    This DBScan clustering component can be applied once to all points
    :math:`\pmb{X} \in \mathbb{R}^{m \times 3}`. Alternatively, it can be
    applied :math:`K \in \mathbb{Z}_{>1}` times. In this last case, consider
    :math:`\pmb{X_1} \in \mathbb{R}^{m_1 \times n_x}, \ldots, \pmb{X_K} \in \mathbb{R}^{m_K \times n_x}`
    as the :math:`K` structure spaces, and compute a DBScan on each of them.
    The :math:`m_k` points in :math:`\pmb{X_k}_{m_k \times n_x}` must represent
    the set of points :math:`\biggl\{{\pmb{x_j*} : y_j = k}\biggr\}`.

    :ivar precluster_name: The name of the attribute to be considered as the
        precluster. If None, then all points will be considered at once instead
        of partitioned by previous clusters.
    :vartype precluster_name: str or None
    :ivar precluster_domain: The domain of the precluster, i.e., the precluster
        labels to be considered. If not given, then any unique precluster
        label will be considered.
    :vartype precluster_domain: list or tuple of str
    :ivar min_points: The minimum number of points in the neighborhood so the
        center point can be considered a kernel point.
    :vartype min_points: int
    :ivar radius: The radius of the neighborhood (typically a spherical
        neighborhood) for spatial queries.
    :vartype raidus: float
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_clustering_args(spec):
        """
        Extract the arguments to initialize/instantiate a DBScanClusterer from
        a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a DBScanClusterer.
        """
        # Extract arguments from parent
        kwargs = Clusterer.extract_clustering_args(spec)
        # Update arguments with those from DBScanClusterer
        kwargs['precluster_name'] = spec.get('precluster_name', None)
        kwargs['precluster_domain'] = spec.get('precluster_domain', None)
        kwargs['min_points'] = spec.get('min_points', None)
        kwargs['radius'] = spec.get('radius', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return kwargs
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of DBScanClusterer.

        :param kwargs: The attributes of the DBScanClusterer that will also
            be passed to the parent.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.precluster_name = kwargs.get('precluster_name', None)
        self.precluster_domain = kwargs.get('precluster_domain', None)
        self.min_points = kwargs.get('min_points', 5)
        self.radius = kwargs.get('radius', 0.5)
        # Validate member attributes
        if self.precluster_domain is not None:
            if not isinstance(self.precluster_domain, (list, tuple)):
                raise ClusteringException(
                    'DBScanClusterer does not support given precluster '
                    f'domain: {self.precluster_domain}'
                )

    # ---  CLUSTERING METHODS  --- #
    # ---------------------------- #
    def fit(self, pcloud):
        """
        The :class:`.DBScanClusterer` does not require any fit at all.
        See :class:`.Clusterer` and :meth:`.Clusterer.fit`.
        """
        return self

    def cluster(self, pcloud):
        """
        Apply DBScan clustering to the given point cloud.

        See :class:`.Clusterer` and :meth:`.Clusterer.cluster`.
        """
        start = time.perf_counter()
        LOGGING.LOGGER.info('Computing DBScan clustering ...')
        # Get structure space
        X = pcloud.get_coordinates_matrix()
        m = X.shape[0]  # Num points
        # Initialize all cluster labels to noise (-1)
        c = np.zeros(m, dtype=int)-1  # Cluster labels
        cluster_idx = 0  # Initial non-noise cluster index (0)
        # Divide in pre-clusters, if requested
        if self.precluster_name is not None:
            # Get pre-clusters
            precluster_low = self.precluster_name.lower()
            if precluster_low == 'classification':
                y = pcloud.get_classes_vector()
            elif precluster_low in ['prediction', 'predictions']:
                y = pcloud.get_predictions_vector()
            else:
                y = pcloud.get_features(self.precluster_name)
            # Determine domain of preclusters
            y_dom = self.precluster_domain
            if y_dom is None:
                y_dom = np.unique(y)
            # Compute a DBScan for each precluster
            for yk in y_dom:
                I = y == yk
                cluster_idx, c[I] = self.do_dbscan(X[I], c[I], cluster_idx)
        # Otherwise, compute all at once
        else:
            cluster_idx, c = self.do_dbscan(X, c, cluster_idx)
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'DBScan clustering computed {cluster_idx} clusters on {m} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return self.add_cluster_labels_to_point_cloud(pcloud, c)

    # ---  DBSCAN METHODS  --- #
    # ------------------------ #
    def do_dbscan(self, X, c, cluster_idx):
        """
        Compute a density-based spatial clustering of applications with noise
        (DBSCAN).

        :param X: The input structure space.
        :type X: :class:`np.ndarray`
        :param c: The vector of point-wise cluster labels for the points in X.
        :type c: :class:`np.ndarray`
        :param cluster_idx: The cluster index for the first cluster.
        :return: The least cluster index greater than the highest cluster index
            assigned to any point.
        :rtype: int
        """
        o3d_cloud = open3d.geometry.PointCloud()
        o3d_cloud.points = open3d.utility.Vector3dVector(X)
        c = cluster_idx + np.array(o3d_cloud.cluster_dbscan(
            self.radius,
            self.min_points,
            print_progress=False
        ), dtype=int)
        return int(np.max(c))+1, c
