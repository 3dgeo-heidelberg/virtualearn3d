# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import joblib
import dill
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

    Moreover, a Gaussian Radial Basis Function (RBF) can be used to
    smooth the features in a given neighborhood such that:

    .. math::

        \hat{f}_i = \dfrac{1}{D} \sum_{j=1}^{\lvert\mathcal{N}\rvert}{
            \exp\left[
                - \dfrac{\lVert{\pmb{x_{i*}} - \pmb{x_{j*}}}\rVert^2}{\omega^2}
            \right] f_j
        }

    Where
    :math:`D = \displaystyle\sum_{j=1}^{\lvert\mathcal{N}\rvert}{\exp\left[-\dfrac{\lVert\pmb{x_{i*}}-\pmb{x_{j*}}\rVert^2}{\omega^2}\right]}`
    .

    One usefult tip to configure a Gaussian RBF with respect to the unitary
    case, i.e., :math:`\exp\left(-\dfrac{1}{\omega^2}\right)` is to define the
    :math:`\omega` parameter of the non-unitary case as
    :math:`\varphi = \sqrt{\omega^2 r^2}` where :math:`r` is the radius of
    the neighborhood. For example, to use a sphere neighborhood of radius 5
    so that a point at 5 meters of the center will have a contribution
    corresponding to a point at one meter in the unitary case is to use
    :math:`\varphi = \sqrt{\omega^2 5^2}` as the new :math:`\omega` for the
    Gaussian RBF.

    :ivar chunk_size: How many points per chunk must be considered when
        computing the data mining in parallel.
    :vartype chunk_size: int
    :ivar subchunk_size: How many neighborhoods per iteration must be
        considered when compting a chunk. It is useful to prevent memory
        exhaustion when considering many big neighborhoods at the same time.
    :vartype subchunk_size: int
    :ivar neighborhood: The definition of the neighborhood to be used. It can
        be a KNN neighborhood:

        .. code-block:: json

            {
                "type": "knn",
                "k": 16
            }

        But it can also be a spherical neighborhood:

        .. code-block:: json

            {
                "type": "sphere",
                "radius": 0.25
            }

    :vartype neighborhood: dict
    :ivar weighted_mean_omega: The :math:`\omega` parameter for the weighted
        mean strategy.
    :vartype weighted_mean_omega: float
    :ivar gaussian_rbf_omega: The :math:`\omega` parameter for the Gaussian
        RBF strategy.
    :vartype gaussian_rbf_omega: float
    :ivar input_fnames: The list with the name of the input features that must
        be smoothed.
    :vartype input_fnames: list
    :ivar fnames: The list with the name of the smooth strategies to be
        computed.
    :vartype fnames: list
    :ivar frenames: The name of the output features.
    :vartype frenames: list
    :ivar nthreads: The number of threads for parallel execution (-1 means
        as many threads as available cores).
    :vartype: nthreads: int
    """

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
            'weighted_mean_omega': spec.get('weighted_mean_omega', None),
            'gaussian_rbf_omega': spec.get('gaussian_rbf_omega', None),
            'input_fnames': spec.get('input_fnames', None),
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
        self.weighted_mean_omega = kwargs.get('weighted_mean_omega', 0.0001)
        self.gaussian_rbf_omega = kwargs.get('gaussian_rbf_omega', 1)
        self.input_fnames = kwargs.get('input_fnames', None)
        self.fnames = kwargs.get(
            'fnames',
            ['mean', 'weighted_mean', 'gaussian_rbf']
        )
        self.frenames = kwargs.get('frenames', None)
        if self.frenames is None:
            neighborhood_type = self.neighborhood['type']
            neighborhood_type_low = neighborhood_type.lower()
            if neighborhood_type_low == 'knn':
                self.frenames = [
                    f'{infname}_{fname}_k{self.neighborhood["k"]}'
                    for fname in self.fnames for infname in self.input_fnames
                ]
            elif neighborhood_type_low == 'sphere':
                self.frenames = [
                    f'{infname}_{fname}_r{self.neighborhood["radius"]}'
                    for fname in self.fnames for infname in self.input_fnames
                ]
        self.nthreads = kwargs.get('nthreads', -1)
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
        neighborhood_type_low = self.neighborhood['type']
        if neighborhood_type_low == 'knn':
            neighborhood_function = self.knn_neighborhood_f
        elif neighborhood_type_low == 'sphere':
            neighborhood_function = self.sphere_neighborhood_f
        else:
            raise MinerException(
                'SmoothFeatsMiner does not support given neighborhood type '
                f'"{self.neighborhood["type"]}".'
            )
        # Determine smooth functions
        smooth_functions = []
        for fname in self.fnames:
            fname_low = fname.lower()
            if fname_low == 'mean':
                smooth_functions.append(self.mean_f)
            elif fname_low == 'weighted_mean':
                smooth_functions.append(self.weighted_mean_f)
            elif fname_low == 'gaussian_rbf':
                smooth_functions.append(self.gaussian_rbf)
            else:
                raise MinerException(
                    'SmoothFeatsMiner was requested to compute an unexpected '
                    f'smooth feature: "{fname}".'
                )
        # Build KDTree
        start = time.perf_counter()
        kdt = dill.dumps(  # Serialized KDT
            KDT(X, leafsize=16, compact_nodes=True, copy_data=False)
        )
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            f'SmoothFeatsMiner built KDTree in {end-start:.3f} seconds.'
        )
        # Chunkify the computation of smooth features
        m = len(X)
        chunk_size = self.chunk_size
        if chunk_size == 0:
            chunk_size = m
        num_chunks = int(np.ceil(m/chunk_size))
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
            for chunk_idx in range(num_chunks)
        )
        # Return point cloud extended with smooth features
        return pcloud.add_features(self.frenames, np.vstack(Fhat))

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
        """
        Compute the smooth features for a given chunk.

        :param X: The structure space matrix (i.e., the matrix of coordinates).
        :param F: The feature space matrix (i.e., the matrix of features).
        :param kdt: The KDTree representing the entire point cloud.
        :param neighborhood_f: The function to extract neighborhoods for the
            points in the chunk.
        :param smooth_funs: The functions to compute the requested smooth
            features.
        :param X_chunk: The structure space matrix of the chunk.
        :param F_chunk: The feature space matrix of the chunk.
        :param chunk_idx: The index of the chunk.
        :return: The smooth features computed for the chunk.
        :rtype: :class:`np.ndarray`
        """
        # Report time for first chunk : start
        if chunk_idx == 0:
            start = time.perf_counter()
        # Compute neighborhoods in chunks (subchunks wrt original problem)
        Fhat_chunk = None
        m = len(X_chunk)
        subchunk_size = self.subchunk_size
        if subchunk_size == 0:
            subchunk_size = m
        num_chunks = int(np.ceil(m/subchunk_size))
        kdt = dill.loads(kdt)  # Deserialized KDTree
        for subchunk_idx in range(num_chunks):
            a_idx = subchunk_idx*subchunk_size  # Subchunk start index
            b_idx = (subchunk_idx+1)*subchunk_size  # Subchunk end index
            X_sub = X_chunk[a_idx:b_idx]  # Subchunk coordinates
            I = neighborhood_f(kdt, X_sub)  # Neighborhood indices
            Fhat_sub = np.hstack([  # Compute smooth features for subchunk
                smooth_f(X, F, X_sub, I) for smooth_f in smooth_funs
            ])
            # Merge subchunk smooth features with chunk smooth features
            if Fhat_chunk is None:
                Fhat_chunk = Fhat_sub
            else:
                Fhat_chunk = np.vstack([Fhat_chunk, Fhat_sub])
        # Report time for first chunk : end
        if chunk_idx == 0:
            end = time.perf_counter()
            print(  # LOGGER cannot be used in multiprocessing contexts
                f'\n\nSmoothFeatsMiner computes a chunk of {F_chunk.shape[0]} '
                f'points with {F_chunk.shape[1]} input features and '
                f'{len(smooth_funs)*F_chunk.shape[1]} output features in '
                f'{end-start:.3f} seconds.\n\n'
            )
        # Return smooth features for input chunk
        return Fhat_chunk

    def mean_f(self, X, F, X_sub, I):
        """
        Mine the smooth features using the mean.

        :param X: The matrix of coordinates representing the input point cloud.
        :param F: The matrix of features representing the intput point cloud.
        :param X_sub: The matrix of coordinates representing the subchunk which
            smooth features must be computed.
        :param I: The list of lists of indices such that the i-th list contains
            the indices of the points in X that belong to the neighborhood
            of the i-th point in X_sub.
        :return: The smooth features for the points in X_sub.
        """
        Fhat = []
        for Ii in I:
            Fhat.append(np.mean(F[Ii], axis=0))
        return Fhat

    def weighted_mean_f(self, X, F, X_sub, I):
        """
        Mine the smooth features using the weighted mean.

        For the parameters and the return see
        :meth:`smooth_feats_miner.SmoothFeatsMiner.mean_f` because
        the parameters and the return are the same but computed with a
        different strategy.
        """
        Fhat = []
        for i, x_sub in enumerate(X_sub):
            J = I[i]
            d = np.linalg.norm(X[J]-x_sub, axis=1)
            dmax = np.max(d)
            d = dmax - d + self.weighted_mean_omega
            D = np.sum(d)
            Fhat.append(np.sum((F[J].T*d).T, axis=0) / D)
        return Fhat

    def gaussian_rbf(self, X, F, X_sub, I):
        """
        Mine the smooth features using the Gaussian Radial Basis Function.

        For the parameters and the return see
        :meth:`smooth_feats_miner.SmoothFeatsMiner.mean_f` because
        the parameters and the return are the same but computed with a
        different strategy.
        """
        Fhat = []
        omega_squared = self.gaussian_rbf_omega*self.gaussian_rbf_omega
        for i, x_sub in enumerate(X_sub):
            J = I[i]
            d = np.exp(-np.sum(np.square(X[J]-x_sub), axis=1)/omega_squared)
            D = np.sum(d)
            Fhat.append(np.sum((F[J].T*d).T, axis=0) / D)
        return Fhat

    # ---  NEIGHBORHOOD FUNCTIONS  --- #
    # -------------------------------- #
    def knn_neighborhood_f(self, kdt, X_sub):
        """
        The k nearest neighbors (KNN) neighborhood function.

        :param kdt: The KDT representing the entire point cloud (X).
        :param X_sub: The points whose neighborhoods must be found.
        :return: The k indices of the nearest neighbors in X for each point in
            X_sub.
        """
        return kdt.query(
            x=X_sub,
            k=self.neighborhood['k'],
            workers=1
        )[1]

    def sphere_neighborhood_f(self, kdt, X_sub):
        """
        The spherical neighborhood function.

        :param kdt: The KDT representing the entire point cloud (X)
        :param X_sub: The points whose neighborhoods must be found.
        :return: The indices of the points in X that belong to the spherical
            neighborhood for each point in X_sub.
        """
        return KDT(X_sub).query_ball_tree(
            other=kdt,
            r=self.neighborhood['radius']
        )
