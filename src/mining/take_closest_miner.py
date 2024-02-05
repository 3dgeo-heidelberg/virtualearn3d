# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import time


class TakeClosestMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena

    Take closest miner.
    See :class:`.Miner`.

    The take closest miner considers a pool of point clouds and for each point
    in the input point cloud takes the requested features from the closest
    neighbor in the entire pool. It can be useful, for example, to have
    a set of mined point clouds and take just some points for training that
    have been manually labeled in the non mined point clouds (e.g., uncertainty
    point clouds, see :class:`.ClassificationUncertaintyEvaluator`).

    :ivar fnames: The names of the features that must be taken from the closest
        neighbor in the pool.
    :vartype fnames: list of str
    :ivar frenames: The names of the features in the output point cloud. When
        not given (i.e., None), they will be the feature names.
    :vartype frenames: list of str
    :ivar y_default: The default value for classification labels. If not given,
        it is the max value for integers ``np.iinfo(int).max``.
    :vartype y_default: int
    :ivar pcloud_pool: The list of paths to the point clouds composing the
        pool.
    :vartype pcloud_pool: list of str
    :ivar distance_upper_bound: The max supported distance. It can be used
        to prune tree searches to speed-up the computations.
    :vartype distance_upper_bound: float
    :ivar nthreads: The number of threads for the parallel closest neighbors
        query. Using -1 implies considering as many threads as available
        cores.
    :vartype nthreads: int
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a TakeClosestMiner from
        a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a TakeClosestMiner.
        """
        # Initialize
        kwargs = {
            'fnames': spec.get('fnames', None),
            'frenames': spec.get('frenames', None),
            'y_default': spec.get('y_default', None),
            'pcloud_pool': spec.get('pcloud_pool', None),
            'distance_upper_bound': spec.get('distance_upper_bound', None),
            'nthreads': spec.get('nthreads', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of TakeClosestMiner.

        :param kwargs: The attributes for the TakeClosestMiner that will also
            be passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the TakeClosestMiner
        self.fnames = kwargs.get('fnames', None)
        self.frenames = kwargs.get('frenames', None)
        self.y_default = kwargs.get('y_default', np.iinfo(int).max)
        self.pcloud_pool = kwargs.get('pcloud_pool', None)
        self.distance_upper_bound = kwargs.get('distance_upper_bound', np.inf)
        self.nthreads = kwargs.get('nthreads', -1)
        if isinstance(self.pcloud_pool, str):
            self.pcloud_pool = [self.pcloud_pool]
        # Validate attributes
        if self.fnames is None:
            raise MinerException(
                'TakeClosestMiner cannot be computed without feature names.'
            )
        if self.frenames is not None:
            if len(self.fnames) != len(self.frenames):
                raise MinerException(
                    f'TakeClosestMiner received {len(self.fnames)} feature '
                    f'names but {len(self.frenames)} feature renames. '
                    'The must be the same number. Alternatively, frenames '
                    'can be None so fnames will be used instead.'
                )
        if self.pcloud_pool is None:
            raise MinerException(
                'TakeClosestMiner cannot be computed without a pool of '
                'point clouds.'
            )

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine feature from closest neighbor in pool.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with taken features.
        :rtype: :class:`.PointCloud`
        """
        # Obtain coordinates
        X = pcloud.get_coordinates_matrix()
        # Prepare fnames
        fnames = []
        take_classes = False
        for fname in self.fnames:
            if fname.lower() == 'classification':
                take_classes = True
            else:
                fnames.append(fname)
        if len(fnames) < 1:
            fnames = None
        # Initialize distances, features, and labels matrices and vectors
        D = np.empty(X.shape[0], dtype=float)
        D.fill(np.inf)
        F = np.empty((X.shape[0], len(fnames)), dtype=float)
        F.fill(np.nan)
        y = np.empty(X.shape[0], dtype=int)
        y.fill(self.y_default)
        # Find features from closest neighbor in pool
        for pcloud_path in self.pcloud_pool:
            # Read input point cloud
            start = time.perf_counter()
            pcloud_i = PointCloudFactoryFacade.make_from_file(pcloud_path)
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'TakeClosestMiner read point cloud at "{pcloud_path}" '
                f'with {pcloud_i.get_num_points()} points in {end-start:.3f} '
                'seconds.'
            )
            # Extract coordinates and features
            X_i = pcloud_i.get_coordinates_matrix()
            F_i = pcloud_i.get_features_matrix(fnames) if fnames is not None \
                else None
            y_i = pcloud_i.get_classes_vector() if take_classes else None
            # Build the KDTree
            start = time.perf_counter()
            kdt = KDT(X_i, leafsize=16)
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'TakeClosestMiner built KDTree in {end-start:.3f} seconds.'
            )
            # Ωuery the KDTree
            start = time.perf_counter()
            D_i, I_i = kdt.query(
                X,
                k=1,
                distance_upper_bound=self.distance_upper_bound,
                workers=self.nthreads
            )
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'TakeClosestMiner queried KDTree in {end-start:.3f} seconds.'
            )
            # Update distances
            mask = D_i < D
            D[mask] = D_i[mask]
            # Remove missing indices
            miss_idx = len(X_i)
            valid_indices = np.array(I_i) != miss_idx
            if np.any(valid_indices):  # If at least one index is non-missing
                I_i = I_i[valid_indices]
                valid_mask = mask[valid_indices]
                # Update for any neigh. that is closer than previous closest
                if fnames is not None:
                    F[mask] = F_i[I_i][valid_mask]
                if take_classes:
                    y[mask] = y_i[I_i][valid_mask]
        # Return point cloud with taken features
        if take_classes:
            pcloud = pcloud.set_classes_vector(y)
        if fnames is not None:
            frenames = fnames
            if self.frenames is not None:
                frenames = self.frenames
            pcloud = pcloud.add_features(frenames, F)
        return pcloud







