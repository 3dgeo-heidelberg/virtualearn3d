# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import numpy as np
import pdal
import json


# ---   CLASS   --- #
# ----------------- #
class CovarFeatsMiner(Miner):
    """
    :author: Hannah Weiser
    :author: Alberto M. Esmoris Pena

    Basic covariance features miner.
    See :class:`.Miner`

    :ivar neighborhood: The type of neighborhood, either "knn" or "spherical".
        Default is "spherical".
    :vartype neighborhood: str
    :ivar mode: How features are computed. Use "SQRT" to consider the square
        root of the eigenvalues, "Normalized" to normalize the eigenvalues
        so they sum to one, or "Raw" to directly use the raw eigenvalues.
        The default mode is "Raw".
    :vartype mode: str
    :ivar nthreads: The number of threads to be used for the parallel
        computation of the covariance features. Default is 1.
    :vartype nthreads: int
    :ivar radius: The radius for the spherical neighborhood (default 0.3).
    :vartpe radius: float
    :ivar min_neighs: The minimum number of neighborhoods (must be given for
        both spherical and knn neighborhoods).
    :vartype min_neighs: int
    :ivar optimize: True to enable optimal neighborhood configuration.
    :vartype optimized: bool
    :ivar fnames: The list of feature names (fnames) attribute.
        ['Density', 'Dimensionality'] by default.
    :vartype fnames: list
    :ivar frenames: Optional attribute to specify how to rename the mined
        features.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a CovarFeatsMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a CovarFeatsMiner.
        """
        # Initialize
        kwargs = {
            'neighborhood': spec.get('neighborhood', None),
            'radius': spec.get('radius', None),
            'min_neighs': spec.get('min_neighs', None),
            'mode': spec.get('mode', None),
            'optimize': spec.get('optimize', None),
            'fnames': spec.get('fnames', None),
            'frenames': spec.get('frenames', None),
            'nthreads': spec.get('nthreads', None),
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Check if fnames contains all and something else
        if 'all' in kwargs['fnames'] and len(kwargs['fnames']) > 1:
            kwargs['fnames'] = ['all']
            LOGGING.LOGGER.info(
                "CovarFeatsMiner received 'all' together with more features.\n"
                "'all' means all features are computed. "
                "No need to specify further features."
            )
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of CovarFeatsMiner.

        :param kwargs: The attributes for the CovarFeatsMiner that will also
            be passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the CovarFeatsMiner
        self.neighborhood = kwargs.get("neighborhood", "spherical").lower()
        self.radius = kwargs.get("radius", 0.3)
        self.mode = kwargs.get("mode", "Raw")
        self.fnames = kwargs.get("fnames", ["Density", "Dimensionality"])
        self.nthreads = kwargs.get("nthreads", -1)
        self.min_neighs = kwargs.get("min_neighs", None)
        self.optimize = kwargs.get("optimize", False)
        # Optional attribute to rename the computed features
        self.frenames = kwargs.get("frenames", None)
        if self.frenames is None:
            if self.neighborhood == "knn":
                self.frenames = [
                   fname+f'_K{self.min_neighs}' for fname in self.fnames
                ]
            elif self.neighborhood == "spherical":
                self.frenames = [
                    fname+f'_r{self.radius}' for fname in self.fnames
                ]

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine covariance features from the given pcloud.
        See :class:`.Miner` and :meth:`.miner.Miner.mine`

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with covariance features.
        """
        # Obtain coordinates matrix
        X = pcloud.get_coordinates_matrix()
        # Validate coordinates matrix
        if X.shape[1] != 3:
            raise MinerException(
                "Covariance features are only supported for 3D point clouds.\n"
                f'However, a {X.shape[1]}D point cloud was given.'
            )
        # Matrix of coordinates to PDAL format
        X = np.array(
            [tuple(X[i]) for i in range(X.shape[0])],
            dtype=[('X', float), ('Y', float), ('Z', float)]
        )
        # Compute covariance features
        pdal_pipeline = self.build_pdal_pipeline(X)
        pdal_pipeline.execute()
        start_feat_idx = 5 if self.needs_optimal_neighborhood() else 3
        feats = np.array([
            pdal_pipeline.arrays[0][name]
            for name in pdal_pipeline.arrays[0].dtype.names[start_feat_idx:]]
        ).T
        # Return point cloud extended with covariance features
        return pcloud.add_features(self.frenames, feats)

    # ---   PDAL METHODS   --- #
    # ------------------------ #
    def build_pdal_pipeline(self, X):
        """
        Build the PDAL pipeline to mine requested covariance features.

        :param X: The matrix of 3D coordinates.
        :return: Built PDAL pipeline.
        """
        # Common specification
        spec = {
            "type": "filters.covariancefeatures",
            "threads": self.nthreads,
            "feature_set": self.fnames,
            "mode": self.mode,
            "optimized": self.optimize
        }
        # If "all" in self.fnames then use "all" instead of list
        if "all" in self.fnames:
            spec['feature_set'] = "all"
        # Handle different neighborhoods
        if self.neighborhood == "spherical":
            spec['radius'] = self.radius
            spec['min_k'] = self.min_neighs
        elif self.neighborhood == "knn":
            spec['knn'] = self.min_neighs
        else:
            raise MinerException(
                f'Unexpected neighborhood: "{self.neighborhood}"'
            )
        # Specification as list
        spec = [spec]
        # Handle optimal neighborhood if necessary
        if self.needs_optimal_neighborhood():
            spec.insert(0, {"type": "filters.optimalneighborhood"})
        # Return built pipeline from specification
        pdal_json = json.dumps(spec)
        LOGGING.LOGGER.debug(f'PDAL JSON specification:\n{pdal_json}')
        return pdal.Pipeline(pdal_json, arrays=[X])

    # ---  CHECKS  --- #
    # ---------------- #
    def needs_optimal_neighborhood(self):
        """
        Checks whether it is necessary to precompute the optimal neighborhood
        before the covariance features or not.

        :return: True if optimal neighborhood is necessary, False otherwise.
        :rtype: bool
        """
        return self.optimize or \
            "Density" in self.fnames or \
            "all" in self.fnames
