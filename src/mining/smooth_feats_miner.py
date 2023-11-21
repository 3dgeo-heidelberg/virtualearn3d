# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class SmoothFeatsMiner(Miner):
    r"""
    :author: Alberto M. Esmoris Pena

    Basic smooth features miner.
    See :class:`.Miner`.

    The smooth features miner considers each point in the point cloud
    :math:`\pmb{x_{i*}}` and finds either each knn or its spherical
    neighborhood :math:`\mathcal{N}`. Now, let :math:`j` index the points
    in the neighborhood. For then, a given feature :math:`f` can be
    smoothed by considering all the points in the neighborhood. In the most
    simple way, the smoothed feature :math:`\hat{f}` can be computed as a mean:

    .. math::

        \hat{f}_i = \dfrac{1}{\lvert\mathcal{N}\rvert}
            \sum_{j=1}^{\lvert\mathcal{N}\rvert}{f_j}

    Alternatively, the feature can be smoothed considering a weighted mean
    where the closest points with respect to :math:`\pmb{x_{i*}}` have a
    greater weight, such that:

    .. math::

        \hat{f}_i = \dfrac{1}{D}\sum_{j=1}^{\lvert\mathcal{N}\rvert}{d_j f_j}

    Where
    :math:`d^*=\max_{j} \; \left\{\lVert\pmb{x_{i*}} - \pmb{x_{j*}}\rVert : j = 1,\ldots,\lvert\mathcal{N}\rvert \right\}`,
    :math:`d_j = d^* - \lVert{\pmb{x_{i*}}-\pmb{x_{j*}}}\rVert + \omega`, and
    :math:`D = \sum_{j=1}^{\mathcal{N}}{d_j}`.

    Moreover, a Radial Basis Function (RBF) Gaussian kernel can be used to
    smooth the features in a given neighborhood such that:

    .. math::

        \hat{f}_i = \dfrac{1}{D} \sum_{j=1}^{\lvert\mathcal{N}\rvert}{
            \exp\left[
                - \dfrac{\lVert{\pmb{x_{i*}} - \pmb{x_{j*}}}\rVert^2}{\omega^2}
            \right]
        }

    Where
    :math:`D = \displaystyle\sum_{j=1}^{\lvert\mathcal{N}\rvert}{\exp\left[-\dfrac{\lVert\pmb{x_{i*}}-\pmb{x_{j*}}\rVert^2}{\omega^2}\right]}`
    .

    """
    # TODO Rethink : Doc attributes (ivar and vartype)

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a SmoothFeatsMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a SmoothFeatsMiner.
        """
        # Initialize
        kwargs = {
            'chunk_size': spec.get('chunk_size', None),
            'subchunk_size': spec.get('subchunk_size', None),
            'neighborhood': spec.get('neighborhood', None),
            'omega': spec.get('omega', None),
            'infnames': spec.get('infnames', None),
            'fnames': spec.get('fnames', None),
            'frenames': spec.get('frenames', None),
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
        Initialize an instance of SmoothFeatsMiner.

        The neighborhood definition and feature names (fnames) are always
        assigned during initialization. The default neighborhood is a knn
        neighborhood with :math:`k=16`.

        :param kwargs: The attributes for the SmoothFeatsMiner that will also
            be passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the SmoothFeatsMiner
        self.chunk_size = kwargs.get('chunk_size', 8000)
        self.subchunk_size = kwargs.get('subchunk_size', 64)
        self.neighborhood = kwargs.get('neighborhood', {
            'type': 'knn',
            'k': 16
        })
        self.omega = kwargs.get('omega', 1)
        self.input_fnames = kwargs.get('input_fnames', None)
        self.fnames = kwargs.get(
            'fnames',
            ['mean', 'mean_weighted', 'gaussian_rbf']
        )
        self.frenames = kwargs.get('frenames', None)
        if self.frenames is None:
            neighborhood_type = self.neighborhood['type']
            neighborhood_type_low = neighborhood_type.lower()
            if neighborhood_type_low == 'knn':
                self.frenames = [
                    f'{fname}_k{self.neighborhood["k"]}'
                    for fname in self.fnames
                ]
            elif neighborhood_type_low == 'sphere':
                self.frenames = [
                    f'{fname}_r{self.neighborhood["radius"]}'
                    for fname in self.fnames
                ]
        self.nthreads=  kwargs.get('nthreads', -1)
        # Validate attributes
        if self.input_fnames is None:
            raise MinerException(
                'SmoothFeatsMiner cannot be built without input features.'
            )

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine smooth features from the given point cloud.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with smooth features.
        :rtype: :class:`.PointCloud`
        """
        # Obtain coordinates and features
        X = pcloud.get_coordinates_matrix()
        F = pcloud.get_features_matrix(self.input_fnames)
        # Determine neighborhood function
        # TODO Rethink : Implement
        neighborhood_function = None
        # Determine smooth functions
        # TODO Rethink : Implement
        smooth_functions = []
        # Build KDTree
        start = time.perf_counter()
        kdt = KDT(X, leafsize=16, compact_nodes=True, copy_data=False)
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            f'SmoothFeatsMiner built KDTree in {end-start:.3f} seconds.'
        )
        # Chunkify the computation of smooth features
        m = len(X)
        chunk_size = self.chunk_size
        if chunk_size == 0:
            chunk_size = m
        LOGGING.LOGGER.debug(
            f'SmoothFeatsMiner computing {int(np.ceil(m/chunk_size))} chunks '
            f'of {chunk_size} points each for a total of {m} points ...'
        )
        Fhat = joblib.Parallel(n_jobs=self.nthreads)(joblib.delayed(
            self.compute_smooth_features
        )(
            X,
            F,
            kdt,
            neighborhood_function,
            smooth_functions,
            X[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size],
            F[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size],
            chunk_idx
        )
            for chunk_idx in range(0, m, chunk_size)
        )
        # Return point cloud extended with smooth features
        return pcloud.add_featres(self.frenames, Fhat)

    # ---  SMOOTH FEATURES METHODS  --- #
    # --------------------------------- #
    def compute_smooth_features(
        self,
        X,
        F,
        kdt,
        neighborhood_f,
        smooth_funs,
        X_chunk,
        F_chunk,
        chunk_idx
    ):
        # TODO Rethink : Doc
        # Report time for first chunk : start
        if chunk_idx == 0:
            start = time.perf_counter()
        # Compute neighborhoods in chunks (subchunks wrt original problem)
        Fhat_chunk = []
        m = len(X_chunk)
        subchunk_size = self.subchunk_size
        if subchunk_size == 0:
            subchunk_size = m
        num_chunks = int(np.ceil(m/subchunk_size))
        for subchunk_idx in range(num_chunks):
            a_idx = subchunk_idx*subchunk_size  # Subchunk start index
            b_idx = (subchunk_idx+1)*subchunk_size  # Subchunk end index
            X_sub = X_chunk[a_idx:b_idx]  # Subchunk coordinates
            I = neighborhood_f(kdt, X_sub)  # Neighborhood indices
            #I = KDT(X_sub).query_ball_tree(kdt)  # TODO Rethink : Sphere neighborhood
            Fhat_sub = np.array([  # Subchunk smooth features
                smooth_f(X, F, X_sub, I) for smooth_f in smooth_funs
            ]).T
            # Merge subchunk smooth features with chunk smooth features
            if Fhat_chunk is None:
                Fhat_chunk = Fhat_sub
            else:
                Fhat_chunk = np.vstack([Fhat_chunk, Fhat_sub])
        # Report time for first chunk : end
        if chunk_idx == 0:
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'SmoothFeatsMiner computes a chunk of {F_chunk.shape[0]} '
                f'points with {F_chunk.shape[1]} features in '
                f'{end-start:.3f} seconds.'
            )
        # Return smooth features for input chunk
        return Fhat_chunk

