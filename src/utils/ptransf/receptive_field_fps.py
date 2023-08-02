# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field import ReceptiveField
from scipy.spatial import KDTree as KDT
import numpy as np
import open3d


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldFPS(ReceptiveField):
    """
    # TODO Rethink : Doc
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
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
        # TODO Rethink : Doc
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
        o3d_cloud = open3d.geometry.PointCloud()
        o3d_cloud.points = open3d.utility.Vector3dVector(X)
        if self.fast:
            step = X.shape[0] // self.num_points
            o3d_cloud = o3d_cloud.uniform_down_sample(step)
        o3d_cloud = o3d_cloud.farthest_point_down_sample(self.num_points)
        self.Y = np.asarray(o3d_cloud.points)
        # Find the indexing matrix N
        kdt = KDT(X)
        self.N = kdt.query(self.Y, k=self.num_encoding_neighbors)[1]
        # Find the indexing matrix M
        kdt = KDT(self.Y)
        self.M = kdt.query(X, k=self.num_encoding_neighbors)[1]
        # Return self for fluent programming
        return self

    def centroids_from_points(self, X):
        """
        # TODO Rethink: Doc
        """
        return self.Y

    def propagate_values(self, v, reduce_strategy='mean'):
        """
        # TODO Rethink: Doc
        """
        # Determine the dimensionality of each value (both scalar and vectors
        # can be propagated). All values must have the same dimensionality.
        try:
            val_dim = len(v[0])
        except Exception as ex:
            val_dim = 1
        # Prepare output matrix
        Ytype = v.dtype if isinstance(v, np.ndarray) else type(v[0])
        Y = np.full([len(self.M), val_dim], np.nan, dtype=Ytype)
        # Populate output matrix : Reduce by mean
        if reduce_strategy == 'mean':
            for i, Mi in enumerate(self.M):
                Y[i] += np.mean(v[Mi])
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
        """
        # TODO Rethink : Doc
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
        # TODO Rethink : Doc
        """
        return X - self.x

    def undo_center_and_scale(self, X):
        """
        # TODO Rethink : Doc
        """
        return X + self.x
