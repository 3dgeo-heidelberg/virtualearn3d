# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
import jakteristics


# ---   CLASS   --- #
# ----------------- #
class GeomFeatsMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena
    :brief: Basic geometric features miner.
    See :class:`Miner`
    :ivar radius: The radius (often in meters) attribute. Radius is 0.3 (often
        meters) by default.
    :vartype radius: float
    :ivar fnames: The list of feature names (fnames) attribute.
        ['linearity', 'planarity', 'sphericity] by default.
    :vartype fnames: list
    :ivar nthreads: The number of threads to be used for the parallel
        computation of the geometric features. Note using -1 (default value)
        implies using as many threads as available cores.
    :vartype nthreads: int
    :ivar frenames: Optional attribute to specify how to rename the mined
        features. None by default.
    :vartype frenames: list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of GeomFeatsMiner.
        The radius and feature names (fnames) are always assigned during
            initialization. Their default values are 0.3 and the list
            ['linearity', 'planarity', 'sphericity'], respectively.
        The number of threads (nthreads or n_jobs) is also assigned during
            initialization with a default value of -1 which means use as many
            threads as available cores.
        :param **kwargs: The attributes for the GeomFeatsMiner that will also
            be passed to the parent.
        """
        # Call parent init
        super(Miner).__init__(**kwargs)
        # Basic attributes of the GeomFeatsMiner
        self.radius = kwargs.get("radius", 0.3)
        self.fnames = kwargs.get("fnames", [
            'linearity', 'planarity', 'sphericity'
        ])
        self.nthreads = kwargs.get("nthreads", kwargs.get("n_jobs", -1))
        # Optional attribute to rename the computed features
        self.frenames = kwargs.get("frenames", None)

    # ---  MINER METHODS  --- #
    # ----------------------- #
    def mine(self, pcloud):
        """
        Mine geometric features from the given pcloud.
        See :class:`Miner` and :method:`Miner.mine()`
        :return: The point cloud extended with geometric features.
        """
        # Obtain coordinates matrix
        X = pcloud.get_coordinates_matrix()
        # Validate coordinates matrix
        if X.shape[1] != 3:
            raise MinerException(
                "Geometric features are only supported for 3D point clouds."
            )
        # Compute geometric features
        feats = jakteristics.compute_features(
            X,
            search_radius=self.radius,
            feature_names=self.fnames,
            num_threads=self.nthreads
        )
        # Return point cloud extended with geometric features
        return pcloud.add_features(
            self.fnames if self.frenames is None else self.frenames,
            feats
        )
