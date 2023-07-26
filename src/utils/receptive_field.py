# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
from scipy.spatial import KDTree as KDT


# ---   CLASS   --- #
# ----------------- #
class ReceptiveField:
    # TODO Rethink : Sphinx doc
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a receptive field util.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveField.
        """
        # Assign attributes
        self.cell_size = kwargs.get('cell_size', np.array([0.05, 0.05, 0.05]))
        self.dimensionality = self.cell_size.shape[0]
        self.bounding_radii = kwargs.get('bounding_radii', None)
        if self.bounding_radii is None:
            raise ValueError(
                'ReceptiveField must be instantiated for a given bounding '
                'radii. None were given.'
            )
        self.num_cells = int(np.round(1/np.prod(self.cell_size/2)))
        self.N = None  # The indexing matrix will be created during fit
        self.x = None  # The center point of the receptive field
        self.m = None  # Number of points the receptive field has been fit to

    # ---   MAIN METHODS   --- #
    # ------------------------ #
    def fit(self, X, x):
        # TODO Rethink : Sphinx doc
        # Center and scale the input point cloud
        self.x = x
        X = self.center_and_scale(X)
        # Find the indexing matrix N
        self.N = self.shadow_indexing_matrix_from_points(X)
        # Store the number of points seen during fit
        self.m = X.shape[0]

    def centroids_from_points(self, X, interpolate=False):
        # TODO Rethink : Sphinx doc
        # Center and scale the input point cloud (X)
        X = self.center_and_scale(X)
        # Compute the centroids (Y)
        nanvec = np.full(X.shape[1], np.nan)
        not_nan_flags = np.sum(self.N >= 0, axis=1, dtype=bool)
        Y = np.array([
            np.mean(X[Ni[Ni >= 0]], axis=0) if not_nan_flags[i] else nanvec
            for i, Ni in enumerate(self.N)
        ])
        # Interpolate the centroid of missing cells
        if interpolate:
            # Prepare interpolation
            b = np.ones(self.x.shape[0])  # Max vertex (1, 1, 1)
            a = -b  # Min vertex (-1, -1, -1)
            cells_per_axis = np.ceil((b-a)/self.cell_size).astype(int)
            # Build support points from missing indices (as empty cell centers)
            missing_indices = np.flatnonzero(~not_nan_flags)
            num_steps = np.array([
                np.mod(
                    np.floor(missing_indices / np.prod(cells_per_axis[:j])),
                    cells_per_axis[j]
                ) for j in range(self.dimensionality)
            ]).T
            sup_missing_Y = a + num_steps * self.cell_size
            # Build KDTs
            non_empty_Y = Y[not_nan_flags]
            non_empty_kdt = KDT(non_empty_Y)
            # Obtain neighborhoods
            num_neighs = 3**self.dimensionality-1
            I = non_empty_kdt.query(
                sup_missing_Y, k=num_neighs
            )[1]
            # Filter neighbors with out-of-bounds index
            if non_empty_Y.shape[0] < num_neighs:
                I = [Ii[Ii < non_empty_Y.shape[0]] for Ii in I]
            # Interpolate from (3^n)-1 neighbors (where n is dimensionality)
            for iter, missing_idx in enumerate(missing_indices):
                # One iteration per missing index (missing_idx)
                Y[missing_idx] = np.mean(non_empty_Y[I[iter]], axis=0)
        # Return
        return Y

    def propagate_values(self, v, safe=True):
        # TODO Rethink : Sphinx doc
        # Determine the dimensionality of each value (both scalar and vectors
        # can be propagated). All values must have the same dimensionality.
        try:
            val_dim = len(v[0])
        except Exception as ex:
            val_dim = 1
        # Prepare output matrix (last row is shadow)
        Ytype = v.dtype if isinstance(v, np.ndarray) else type(v[0])
        Y = np.full([self.m+1, val_dim], np.nan, dtype=Ytype)
        # Populate output matrix
        for i, Ni in enumerate(
            self.N[np.sum(self.N >= 0, dtype=bool, axis=1)]  # Non-empty cells
        ):
            Y[Ni] = v[i]
        # Remove shadow row (it was used for shadow neighbors, i.e., index -1)
        Y = Y[:-1]
        # Validate
        if safe:
            if np.any(np.isnan(Y)):
                raise ValueError(
                    'The receptive field propagated NaN values. This is not '
                    'allowed in safe mode.'
                )
        # Return output matrix (or vector if single-column)
        return Y if Y.shape[1] > 1 else Y.flatten()

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def shadow_indexing_matrix_from_points(self, X):
        # TODO Rethink : Sphinx doc
        # Obtain point-wise cell index (I)
        b = np.ones(self.x.shape[0])  # Max vertex (1, 1, 1)
        a = -b  # Min vertex (-1, -1, -1)
        dim_factors = np.ones(self.dimensionality, dtype=int)
        for i in range(1, self.dimensionality):
            dim_factors[i] = int(
                dim_factors[i-1] * np.ceil(
                    (b[i-1]-a[i-1]) / self.cell_size[i-1]
                )
            )
        I = np.sum(np.apply_along_axis(  # Compute point-wise indices
            np.clip,  # Function to be applied
            1,  # Axis 1, i.e., apply along columns (coordinates)
            (  # Compute the coordinate-wise indices for each point
                np.floor((X-a)/self.cell_size + 1e-15) * dim_factors
            ).astype(int),
            None,  # No clip to min, only coordinate-wise clip to max
            ((2/self.cell_size-1)*dim_factors).astype(int)
        ), axis=1)  # Point index as superposition of coordinate indices
        # Populate cells
        max_num_neighs = np.max(np.unique(I[I != -1], return_counts=True)[1])
        N = np.full((self.num_cells, max_num_neighs), -1, dtype=int)
        index_pointers = np.zeros(self.num_cells, dtype=int)
        for j, i in enumerate(I):
            N[i, index_pointers[i]] = j
            index_pointers[i] += 1
        # Return
        return N

    def center_and_scale(self, X):
        # TODO Rethink : Sphinx doc
        return (X - self.x) / self.bounding_radii

    def undo_center_and_scale(self, X):
        # TODO Rethink : Sphinx doc
        return self.bounding_radii * X + self.x
