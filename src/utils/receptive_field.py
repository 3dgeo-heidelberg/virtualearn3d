# ---   IMPORTS   --- #
# ------------------- #
import numpy as np


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
        self.bounding_radius = kwargs.get('bounding_radius', None)
        if self.bounding_radius is None:
            raise ValueError(
                'ReceptiveField must be instantiated for a given bounding '
                'radius. None was given.'
            )
        self.num_cells = int(np.round(
            np.power(2, self.dimensionality, dtype=int) /
            np.prod(self.cell_size)
        ))
        self.N = None  # The indexing matrix will be created during fit
        self.x = None  # The center point of the receptive field
        self.m = None  # Number of points the receptive field has been fit to

    # ---   MAIN METHODS   --- #
    # ------------------------ #
    def fit(self, X, x):
        # TODO Rethink : Sphinx doc
        # Center and scale the input point cloud
        self.x = x
        X = (X - self.x) / self.bounding_radius
        # Find the indexing matrix N
        self.N = self.shadow_indexing_matrix_from_points(X)
        # Store the number of points seen during fit
        self.m = X.shape[0]
        # TODO Rethink : Interpolate missing neighborhoods with 3^n-1

    def centroids_from_points(self, X, interpolate=False):
        # TODO Rethink : Sphinx doc
        # TODO Rethink : Implement
        # Center and scale the input point cloud
        X = (X - self.x) / self.bounding_radius
        # Compute the centroids
        Y = np.array([np.mean(X[Ni[Ni >= 0]], axis=0) for Ni in self.N])
        # Interpolate the centroid of missing cells
        if interpolate:  # TODO Rethink : Implement
            # Prepare interpolation
            b = np.ones(self.x.shape[0])  # Max vertex (1, 1, 1)
            a = -b  # Min vertex (-1, -1, -1)
            cells_per_axis = np.ceil((b-a)/self.cell_size).astype(int)
            # Build support points from missing indices
            missing_indices = np.flatnonzero(np.sum(np.isnan(Y), axis=1))
            num_steps = np.array([
                np.mod(
                    np.floor(missing_indices / np.prod(cells_per_axis[:j])),
                    cells_per_axis[j]
                ) for j in range(self.dimensionality)
            ]).T
            sup_missing_Y = a + num_steps * self.cell_size
            # Interpolate from (3^n)-1 neighbors (where n is dimensionality)
            for i in missing_indices:  # For each empty cell i
                Y[i] = np.mean(Y[I[i]])

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
        Y = np.full([self.m+1, val_dim], np.nan)
        # Populate output matrix
        for i, Ni in enumerate(self.N):
            Y[Ni] = v[i]
        # Remove shadow row
        Y = Y[:-1]
        # Validate
        if safe:
            if np.any(np.isnan(Y)):
                raise ValueError(
                    'The receptive field propagated NaN values. This is not '
                    'allowed in safe mode.'
                )
        # Return output matrix
        return Y

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
                    (b[i]-a[i]) / self.cell_size[i]
                )
            )
        I = (np.floor((X-a) / self.cell_size) * dim_factors).astype(int)
        # Populate cells
        max_num_neighs = np.max(np.unique(I[I!=-1], return_counts=True)[1])
        N = np.full((self.num_cells, max_num_neighs), np.nan, dtype=int)
        index_pointers = np.zeros(self.num_cells, dtype=int)
        for j, i in enumerate(I):
            N[i, index_pointers[i]] = j
            index_pointers[i] += 1
        # Return
        return N
