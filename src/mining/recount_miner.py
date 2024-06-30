# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.mining.smooth_feats_miner import SmoothFeatsMiner
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class RecountMiner(Miner):
    r"""
    :author: Alberto M. Esmoris Pena

    Recount miner.
    See :class:`.Miner`.

    The recount miner considers each point in the point cloud
    :math:`\pmb{x_{i*}}` and finds either the knn or its spherical
    neighborhood :math:`\mathcal{N}`. Now, let :math:`j` index the points
    in the neighborhood. For then, a given feature :math:`f` (or reference
    value :math:`y`, e.g., classification label) can be used to filter the
    points (e.g., selecting :math:`\pmb{x}_{i*} \in \mathcal{N}` such that
    the j-th feature for the i-th points satisfies :math:`{f_i > \tau}`, for
    a given threshold :math:`\tau`. Finally, all the points in the filtered
    neighborhood can be counted in terms of absolute and relative frequency,
    and also with respect to the surface or the volume of the neighborhood
    (given by the radius of the spherical neighborhood or the distance wrt the
    closest nearest neighbor for knn neighborhoods).
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a RecountMiner from a
        key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a RecountMiner.
        """
        # Initialize
        kwargs = {
            'chunk_size': spec.get('chunk_size', None),
            'subchunk_size': spec.get('subchunk_size', None),
            'nthreads': spec.get('nthreads', None),
            'neighborhood': spec.get('neighborhood', None),
            'filters': spec.get('filters', None),
            'input_fnames': spec.get('input_fnames', None)
        }
        # Delete keys with None Value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of RecountMiner.

        The neighborhood definition and feature names (fnames) are always
        assigned during initialization. The default neighborhood is a knn
        neighborhood with :math:`k=16`.

        :param kwargs: The attributes for the RecountMiner that will also be
            passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the RecountMiner
        self.chunk_size = kwargs.get('chunk_size', 8000)
        self.subchunk_size = kwargs.get('subchunk_size', 64)
        self.nthreads = kwargs.get('nthreads', -1)
        self.neighborhood = kwargs.get('neighborhood', {
            'type': 'knn',
            'k': 16
        })
        self.filters = kwargs.get('filters', None)
        self.input_fnames = kwargs.get('input_fnames', None)
        # Validate attributes
        if self.filters is None:
            raise MinerException(
                'RecountMiner did not receive any filter. This is not '
                'supported. At least one filter without conditions must be '
                'given to specify how to count.'
            )
        # Automatically deduce frenames
        self.frenames = []
        for f in self.filters:
            self.frenames.extend(self.get_recount_names_from_filter(f))

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine recounts on filtered neighborhoods from the given point cloud.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with recounts.
        :rtype: :class:`.PointCloud`
        """
        # Obtain coordinates and features
        X = self.get_structure_space_matrix(pcloud)
        F = self.get_feature_space_matrix(pcloud, self.input_fnames)
        # Prepare neighborhood handling and KDTree
        neighborhood_radius, neighborhood_function, kdt = \
            SmoothFeatsMiner.prepare_mining(self, X)
        # Chunkify the recounts
        num_chunks, chunk_size = SmoothFeatsMiner.prepare_chunks(self, X)
        LOGGING.LOGGER.debug(
            f'RecountMiner computing {num_chunks} chunks '
            f'of {chunk_size} points each for a total of {len(X)} points ...'
        )
        # Compute recounts in parallel for each chunk
        Fhat = joblib.Parallel(n_jobs=self.nthreads)(joblib.delayed(
            self.compute_recount
        )(
            X,
            F,
            kdt,
            neighborhood_function,
            neighborhood_radius,
            X[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size],
            F[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size],
            chunk_idx
        )
            for chunk_idx in range(num_chunks)
        )
        # Return point cloud extended with recounts
        return pcloud.add_features(
            self.frenames, np.vstack(Fhat), ftypes=F.dtype
        )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def get_recount_names_from_filter(self, f):
        """
        Obtain the new feature names generated by the filter.

        :param f: The filter specification.
        :type f: dict
        :return: The names of the new features generated by the filter.
        :rtype: list
        """
        filter_name = f['filter_name']
        filter_names = []
        if f.get('absolute_frequency', False):
            filter_names.append(f'{filter_name}_abs')
        if f.get('relative_frequency', False):
            filter_names.append(f'{filter_name}_rel')
        if f.get('surface_density', False):
            filter_names.append(f'{filter_name}_sd')
        if f.get('volume_density', False):
            filter_names.append(f'{filter_name}_vd')
        if f.get('vertical_segments', 0) > 1:
            filter_names.append(f'{filter_name}_vs')
        return filter_names

    def fname_to_feature_index(self, fname):
        """
        Obtain the feature index corresponding to the given fname.

        :param fname: The name of the feature which index must be found.
        :type fname: str
        :return: The index of the feature with the given name.
        :rtype: int
        """
        for i, finame in enumerate(self.input_fnames):
            if fname == finame:
                return i
        raise MinerException(
            'RecountMiner could not find the index of the feature with name '
            f'"{fname}".'
        )

    # ---   RECOUNT METHODS   --- #
    # --------------------------- #
    def compute_recount(
        self,
        X,
        F,
        kdt,
        neighborhood_f,
        neighborhood_radius,
        X_chunk,
        F_chunk,
        chunk_idx
    ):
        """
        Compute the recounts for a given chunk.

        :param X: The structure space matrix (i.e., the matrix of coordinates).
        :param F: The feature space matrix (i.e., the matrix of features).
        :param kdt: The KDTree representing the entire point cloud.
        :param neighborhood_f: The function to extract neighborhoods for the
            points in the chunk.
        :param neighborhood_radius: The radius of the spherical neighborhood or
            None to be computed from the points (e.g., for knn neighborhoods).
        :param X_chunk: The structure space matrix of the chunk.
        :param F_chunk: The feature space matrix of the chunk.
        :param chunk_idx: The index of the chunk.
        :return: The recount features computed for the chunk.
        :rtype: :class:`np.ndarray`
        """
        # Report time for first chunk : start
        if chunk_idx == 0:
            start = time.perf_counter()
        # Prepare chunk
        num_subchunks, subchunk_size, kdt = SmoothFeatsMiner.prepare_chunk(
            self, X_chunk, kdt
        )
        Fhat_chunk = None
        # Compute chunk by subchunks
        for subchunk_idx in range(num_subchunks):
            # Obtain subchunk coordinates and neighborhood indices
            X_sub, I = SmoothFeatsMiner.prepare_subchunk(
                self, subchunk_idx, subchunk_size, X_chunk, kdt, neighborhood_f
            )
            # Compute recount features for the subchunk
            Fhat_sub = np.hstack([
                self.compute_filter(
                    f, X, F, X_sub, list(I), neighborhood_radius
                ) for f in self.filters
            ])
            # Merge subchunk recount features with chunk recount features
            Fhat_chunk = SmoothFeatsMiner.prepare_mined_features_chunk(
                Fhat_chunk, Fhat_sub
            )
        # Report time for first chunk : end
        if chunk_idx == 0:
            end = time.perf_counter()
            print(  # LOGGER cannot be used in multiprocessing contexts
                f'\n\nRecountMiner computes a chunk of {F_chunk.shape[0]} '
                f'points with {F_chunk.shape[1]} input features and '
                f'{Fhat_chunk.shape[1]} output features in '
                f'{end-start:.3f} seconds.\n\n'
            )
        # Return recount features for input chunk
        return Fhat_chunk

    def compute_filter(self, f, X, F, X_sub, I, r):
        """
        Compute the given filter on the neighborhoods of a given chunk.

        :param f: The specification of the filter to be computed.
        :type f: dict
        :param X: The matrix of coordinates representing the input point cloud.
        :type X: :class:`np.ndarray`
        :param F: The matrix of features representing the intput point cloud.
        :type F: :class:`np.ndarray`
        :param X_sub: The matrix of coordinates representing the subchunk which
            recount features must be computed.
        :type X_sub: :class:`np.ndarray`
        :param I: The list of lists of indices such that the i-th list contains
            the indices of the points in X that belong to the neighborhood
            of the i-th point in X_sub.
        :type I: list of list of int
        :param r: The radius of the spherical neighborhood, None if it must be
            computed from the points in the neighborhood.
        :type r: float or None
        :return: The recount features for the points in X_sub.
        :rtype: :class:`np.ndarray`
        """
        # Handle NaNs
        if f.get('ignore_nan', False):
            for i, Ii in enumerate(I):
                # Find NaNs in the i-th neighborhood of the chunk
                Ff = F[Ii]
                nan_indices = ~np.any(np.isnan(Ff), axis=1)
                # Filter out NaNs from the i-th neighborhood
                I[i] = np.array(Ii, dtype=int)[nan_indices].tolist()
        # Compute the recounts for each neighborhood
        all_recounts = []
        for i, Ii in enumerate(I):
            # Get number of points before checking conditions
            total_pts = len(Ii)
            # Apply conditions
            conditions = f.get('conditions', None)
            if conditions is not None:
                for condition in conditions:
                    condition['feature_index'] = self.fname_to_feature_index(
                        condition['value_name']
                    )
                if conditions is not None:
                    Ii = RecountMiner.apply_conditions(Ii, F, conditions)
            # Obtain coordinates and features
            Ff = F[Ii]
            Xf = X[Ii]
            # Do recounts for i-th neighborhood
            recounts = []
            if f.get('absolute_frequency', False):
                recounts.append(self.recount_absolute_frequency(Ff))
            if f.get('relative_frequency', False):
                recounts.append(self.recount_relative_frequency(Ff, total_pts))
            if f.get('surface_density', False):
                recounts.append(self.recount_surface_density(
                    Ff, Xf[:, :2], X_sub[i, :2], r
                ))
            if f.get('volume_density', False):
                recounts.append(self.recount_volume_density(
                    Ff, Xf, X_sub[i], r
                ))
            if f.get('vertical_segments', 0) > 1:
                recounts.append(self.recount_vertical_segments(
                    Ff, Xf[:, 2], f['vertical_segments']
                ))
            # Append recounts for i-th neighborhood
            all_recounts.append(np.hstack(recounts))
        # Return all recounts
        return np.vstack(all_recounts)

    def recount_absolute_frequency(self, F):
        """
        Count the number of points.
        """
        return len(F)

    def recount_relative_frequency(self, F, total_pts):
        """
        The number of points after filtering divided by the total number
        of points before filtering.
        """
        return len(F) / total_pts if total_pts > 0 else 0

    def recount_surface_density(self, F, X2D, x, r):
        r"""
        The number of points after filtering divided by the area of the
        neighborhood.

        If the neighborhood is a spherical one with radius :math:`r` then the
        area will be given by :math:`\pi r^2`. If the neighborhood is a knn
        one then the area will be given by
        :math:`\pi \left(\dfrac{d^*}{2}\right)^2`,
        where :math:`d^*` is the distance between the :math:`(x, y)`
        coordinates of the center point and the furthest one.
        """
        # Compute radius from neighborhood, if necessary
        if r is None:
            r = np.sqrt(np.max(np.sum(np.power(X2D-x, 2), axis=1)))
        # Compute the surface density
        return len(F)/(np.pi*r*r)

    def recount_volume_density(self, F, X, x, r):
        r"""
        The number of points after filtering divided by the volume of the
        neighborhood.

        If the neighborhood is a spherical one with radius :math:`r` then the
        volume will be given by :math:`\dfrac{4}{3}\pi r^2`. If the
        neighborhood is a knn one then the area will be given by
        :math:`\dfrac{4}{3}\pi \left(\dfrac{d^*}{2}\right)^2`, where
        :math:`d^*` is the distance between the :math:`(x, y, z)` coordinates
        of the center point and the furthest one.
        When using a cylinder, the radius will be
        considered to compute the area and the volume will be computed
        considering the vertical boundaries of the cylinder such that
        :math:`\pi r^2 (z^*-z_*)` where :math:`z_*` is the min vertical
        coordinate and :math:`z^*` is the max vertical coordinate.

        Note that, for cylindrical neighborhoods, if there is no difference
        between the max and the min vertical coordinate, then the maximum
        integer will be returned, effectively avoiding a division by zero.
        """
        # Handle no points case
        if len(F) == 0:
            return 0
        # Handle cylindrical neighborhoods
        if self.neighborhood['type'].lower() == 'cylinder':
            z = X[:, 2]
            zmin, zmax = np.min(z), np.max(z)
            zdelta = zmax-zmin
            bounded_cylinder_volume = (np.pi*r*r)
            if zdelta == 0:
                return np.iinfo(int).max
            return len(F)/bounded_cylinder_volume
        # Compute radius from neighborhood, if necessary
        if r is None:
            r = np.sqrt(np.max(np.sum(np.power(X-x, 2), axis=1)))
        # Compute the volume density
        return len(F)/(4.0*np.pi*r*r*r/3.0)

    def recount_vertical_segments(self, F, z, num_segments):
        """
        The number of vertical segments along a vertical cylinder that contain
        at least one point.
        """
        # If no points at all, count will be zero
        if len(F) < 1:
            return 0
        # Find cut points separating vertical segments
        zmin, zmax = np.min(z), np.max(z)
        cuts = np.linspace(zmin, zmax, num_segments)
        # Analyze the first segment
        count = int(np.any(z <= cuts[0]))
        # Analyze following segments
        for i in range(1, num_segments):
            a, b = cuts[i-1], cuts[i]
            count += int(np.any((z > a)*(z <= b)))
        # Return count of vertical segments with at least one point
        return count

    # ---  CONDITION FUNCTIONS  --- #
    # ----------------------------- #
    @staticmethod
    def apply_conditions(I, F, conditions):
        """
        Apply the conditions to filter out all the points that do not satisfy
        one or more of them.

        :param I: The indices for the current neighborhood.
        :type I: list
        :param F: The features
        :type F: :class:`np.ndarray`
        :param conditions: The conditions to be applied.
        :type conditions: list
        :return: The indices of the current neighborhood that satisfy the
            conditions.
        :rtype: list
        """
        # Filter out all the points that dont satisfy at least one condition
        for condition in conditions:
            f_idx = condition['feature_index']
            fI = F[I][:, f_idx]
            Imask = RecountMiner.apply_condition(fI, condition)
            I = np.array(I, dtype=int)[Imask].tolist()
        # Return filtered neighborhood
        return I

    @staticmethod
    def apply_condition(f, condition):
        """
        Check whether the condition is satisfied for each given point.

        :param f: The feature vector where the condition must be checked.
        :param condition: The specification of the condition to be checked.
        :return: The mask with True for points that satisfy the condition,
            False otherwise.
        :rtype: :class:`np.ndarray` of bool
        """
        cond_type = condition['condition_type']
        target = condition['value_target']
        try:
            cond_type_low = condition['condition_type']
            if cond_type_low == 'not_equals':
                return f != target
            elif cond_type_low == 'equals':
                return f == target
            elif cond_type_low == 'less_than':
                return f < target
            elif cond_type_low == 'less_than_or_equal_to':
                return f <= target
            elif cond_type_low == 'greater_than':
                return f > target
            elif cond_type_low == 'greater_than_or_equal_to':
                return f >= target
            elif cond_type_low == 'in':
                return np.array([fi in target for fi in f], dtype=bool)
            elif cond_type_low == 'not_in':
                return np.array([fi not in target for fi in f], dtype=bool)
            else:
                LOGGING.LOGGER.error(
                    'RecountMiner.apply_condition received an unexpected '
                    f'condition_type: "{cond_type}"'
                )
                raise MinerException(
                    f'Unexpected condition type: "{cond_type}"'
                )
        except Exception as ex:
            LOGGING.LOGGER.error(
                'RecountMiner failed to apply condition with type '
                f'"{cond_type}".'
            )
            raise MinerException(
                'RecountMiner failed to apply condition with type '
                f'"{cond_type}".'
            ) from ex

    # ---   DECORATION METHODS   --- #
    # ------------------------------ #
    def get_decorated_fnames(self):
        """
        Obtain the names of the recount features.

        :return: List with the names of the recount features.
        :rtype: list of str
        """
        return self.frenames
