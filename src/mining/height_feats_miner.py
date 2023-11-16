# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.utils.dict_utils import DictUtils
from scipy.spatial import KDTree as KDT
import scipy
import numpy as np
import joblib


# ---   CLASS   --- #
# ----------------- #
class HeightFeatsMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena

    Basic height features miner.
    See :class:`.Miner`.

    :ivar radius: The radius (often in meters) attribute. Radius is 100 by
        default. The radius can describe either the disk of a cylinder or
        half the side of a rectangular region along the vertical axis.
    :vartype radius: float
    :ivar neighborhood: The neighborhood definition. For example:

        .. code-block:: json

            {
                "type": "cylinder",
                "radius": 100,
                "separation_factor": 0.8
            }

    :vartype neighborhood: dict
    :ivar outlier_filter: The outlier filter to be applied (if any).
    :vartype outlier_filter: str or None
    :ivar fnames: The list of height features that must be mined.
        ['floor_distance'] by default.
    :vartype fnames: list
    :ivar nthreads: The number of threads to be used for the parallel
        computation of the geometric features. Note using -1 (default value)
        implies using as many threads as available cores.
    :vartype nthreads: int
    :ivar frenames: Optional attribute to specify how to rename the mined
        features.
    :vartype frenames: list
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a HeightFeatsMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a HeightFeatsMiner.
        """
        # Initialize
        kwargs = {
            'support_chunk_size': spec.get('support_chunk_size', None),
            'neighborhood': spec.get('neighborhood', None),
            'outlier_filter': spec.get('outlier_filter', None),
            'fnames': spec.get('fnames', None),
            'frenames': spec.get('frenames', None),
            'nthreads': spec.get('nthreads', None),
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of HeightFeatsMiner.

        The neighborhood definition and feature names (fnames) are always
        assigned during initialization. The default neighborhood is a cylinder
        with a disk of radius 100 and

        :param kwargs: The attributes for the HeightFeatsMiner that will also
            be passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the HeightFeatsMiner
        self.support_chunk_size = kwargs.get('support_chunk_size', 0)
        self.neighborhood = kwargs.get(
            'neighborhood',
            {
                'type': 'cylinder',
                'radius': 100.0,
                'separation_factor': 0.8
            }
        )
        self.outlier_filter = kwargs.get('outlier_filter', None)
        self.fnames = kwargs.get('fnames', ['floor_distance'])
        self.frenames = kwargs.get('frenames', None)
        self.nthreads = kwargs.get('nthreads', -1)
        if self.frenames is None:
            self.frenames = [
                fname +
                f'_r{self.neighborhood["radius"]}' +
                f'_r{self.neighborhood["separation_factor"]}'
                for fname in self.fnames
            ]

    # ---  MINER METHODS  --- #
    # ----------------------- #
    def mine(self, pcloud):
        """
        Mine height features from the given point cloud.
        See :class:`.Miner` and :meth:`mining.Miner.mine`.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with height features.
        :rtype: :class:`.PointCloud`
        """
        # Obtain coordinates matrix
        X = pcloud.get_coordinates_matrix()
        # Validate coordinates matrix
        if X.shape[1] != 3:
            raise MinerException(
                'Height features are only supported for 3D point clouds.\n'
                f'However, a {X.shape[1]}D point cloud was given.'
            )
        # Compute height features
        feats = self.compute_height_features(X)
        # Return point cloud extended with height features
        return pcloud.add_features(self.frenames, feats)

    # ---  HEIGHT FEATURES METHODS  --- #
    # --------------------------------- #
    def compute_height_features(self, X):
        r"""
        Compute the height features for the given matrix of coordinates
        :pmb:`\mathbb{X} \in \mathbb{R}^{m \times 3}`.

        :param X: The matrix of coordinates.
        :type X: :class:`np.ndarray`
        :return: The computed features.
        :rtype: :class:`np.ndarray`
        """
        # Compute support points
        sup_X = GridSubsamplingPreProcessor.build_support_points(
            X=X,
            separation_factor=self.neighborhood['separation_factor'],
            sphere_radius=self.neighborhood['radius'],
            center_on_X=False,
            support_strategy='grid',
            nthreads=self.nthreads
        )
        # Compute height features for each support neighborhood
        kdt = KDT(X[:, :2])
        sup_feats = self.compute_height_features_on_support(X, sup_X, kdt)
        # Propagate support features to point cloud
        kdt = KDT(sup_X)
        feats = self.compute_pwise_height_features(X, sup_X, sup_feats, kdt)
        # Return point-wise height features
        return feats

    def compute_height_features_on_support(self, X, sup_X, kdt):
        """
        Compute the height features on each support neighborhood.

        :param X: The matrix of coordinates representing the input point cloud.
        :param sup_X: The center point for each support neighborhood.
        :param kdt: The KDTree representing the input point cloud on (x, y)
            only (i.e., 2D).
        :return: The height features for each support point.
        :rtype: :class:`np.ndarray`
        """
        # Function to compute height features for a given support neighborhood
        def compute_pwise_support_feats(z, height_functions):
            # Outlier filtering
            if self.outlier_filter is not None:
                filter_low = self.outlier_filter.lower()
                if filter_low == 'iqr':
                    Q = np.quantile(z, [0.25, 0.75])
                    IQR = Q[1]-Q[0]
                    zmin, zmax = Q[0]-1.5*IQR, Q[1]+1.5*IQR
                    z = z[(z >= zmin) * (z <= zmax)]
                elif filter_low == 'stdev':
                    mean, stdev = np.mean(z), np.std(z)
                    zmin, zmax = mean-3*stdev, mean+3*stdev
                    z = z[(z >= zmin) * (z <= zmax)]
                else:
                    raise MinerException(
                        'HeightFeatsMiner does not support the requested '
                        f'outlier filter "{self.outlier_filter}".'
                    )
            # Return
            return np.array([height_f(z) for height_f in height_functions])
        # Prepare chunk strategy
        num_chunks, chunk_size = 1, len(sup_X)
        if self.support_chunk_size > 0:
            chunk_size = self.support_chunk_size
            num_chunks = int(np.ceil(len(sup_X)))
        height_functions = self.select_height_functions()
        F = []
        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Extract chunk
            sup_idx_a = chunk_idx * chunk_size
            sup_idx_b = min((chunk_idx+1)*chunk_size, len(sup_X))
            chunk_sup_X = sup_X[sup_idx_a:sup_idx_b, :2]
            # Find neighbors of support points in point cloud
            I = KDT(chunk_sup_X[:, :2]).query_ball_tree(
                kdt, self.neighborhood['radius']
            )
            # Compute height features for each neighborhood
            Z = [X[Ii][:, 2] for Ii in I]
            F = F + joblib.Parallel(n_jobs=self.nthreads)(
                joblib.delayed(compute_pwise_support_feats)(
                    Zi, height_functions
                )
                for Zi in Z
            )
        # Return
        return np.array(F)

    def compute_pwise_height_features(self, X, sup_X, sup_F, kdt):
        """
        Compute the height features for each point in the point cloud.

        :param X: The matrix of coordinates representing the input point cloud.
        :param sup_X: The center point for each support neighborhood.
        :param sup_F: The features for each support point.
        :param kdt: The KDTree representing the support points.
        :return: The height features for each point in the point cloud.
        :rtype: :class:`np.ndarray`
        """
        # TODO Rethink : Implement
        # Find neighborhoods of X in support
        I = kdt.query(X, k=1)[1]

    # ---  UTIL FUNCTIONS  --- #
    # ------------------------ #
    def select_height_functions(self):
        """
        Select height functions from specified feature names (fnames). Always
        includes the computation of the floor and ceil coordinates.

        :return: List of functions to extract height features from a vector of
            vertical coordinates. Each feature is a map from a vector of
            arbtirary dimensionality representing height coordinates to a
            single scalar.
        :rtype: list
        """
        height_functions = [np.min, np.max]
        for fname in self.fnames:
            fname_low = fname.lower()
            if fname_low == 'height_range':
                height_functions.append(lambda z: np.max(z)-np.min(z))
            elif fname_low == 'mean_height':
                height_functions.append(np.mean)
            elif fname_low == 'median_height':
                height_functions.append(np.median)
            elif fname_low == 'height_quartiles':
                height_functions.append(
                    lambda z: np.quantile(z, [0.25, 0.5, 0.75])
                )
            elif fname_low == 'height_deciles':
                height_functions.append(
                    lambda z: np.quantile(z, [(i+1)/10 for i in range(9)])
                )
            elif fname_low == 'height_variance':
                height_functions.append(np.var)
            elif fname_low == 'height_stdev':
                height_functions.append(np.std)
            elif fname_low == 'height_skewness':
                height_functions.append(scipy.stats.skew)
            elif fname_low == 'height_kurtosis':
                height_functions.append(scipy.stats.kurtosis)
            elif fname_low not in [
                'floor_coordinate',
                'floor_distance',
                'ceil_coordinate',
                'ceil_distance'
            ]:
                raise MinerException(
                    f'HeightFeatsMiner does not support the feature "{fname}".'
                )
        return height_functions
