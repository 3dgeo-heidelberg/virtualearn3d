# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field import ReceptiveField
import numpy as np
from scipy.spatial import KDTree as KDT


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldGS(ReceptiveField):
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing a receptive field based on grid subsampling.

    Receptive fields constitute a discrete representation of a point cloud of
    m points using R points. More formally, the receptive field can be
    defined as a transformation :math:`\mathcal{R}(\pmb{X})` such that:

    .. math::
        \mathcal{R}:
            \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{R \times n}

    The receptive field transformation is modelled using an indexing matrix
    :math:`\pmb{N} \in \mathbb{Z}_{\geq -1}^{R \times R^*}`, where :math:`R^*`
    represents the number of correspondences of the point in
    :math:`\mathcal{R}` that corresponds to the maximum number of points in
    :math:`\pmb{X}`. Each row
    :math:`\pmb{n}_{i*} \in \mathbb{Z}_{\geq -1}^{R^*}` contains the indices
    of the points in :math:`\pmb{X}` that are associated to the point :math:`i`
    in :math:`\mathcal{R}`. Furthermore, the greater-than-or-equal-to-zero
    integer universe is extended with a :math:`-1` symbol that does not
    represent the minus one number, but instead it is the shadow element of
    the universe. This shadow index is interpreted in the context of the
    receptive field as a mask that disables certain computations for some empty
    cells in the receptive field. It is used to model :math:`\pmb{N}` as a
    matrix, for convenience.

    The baseline receptive field is implemented as an n-dimensional grid
    that supports axis-wise steps for its definition. It can be used to build
    a representation of a point cloud, do some computation on the
    representation, and then propagate the values back to the original point
    cloud. For example, the computation can be a classification computed by
    a neural network that works on sets of :math:`R` points.

    See :class:`.ReceptiveField` and :class:`.GridSubsamplingPreProcessor`.

    :ivar cell_size: The cell size with respect to the radius. It is noted as
        :math:`\pmb{s} = (s_1, \ldots, s_n)`. Each :math:`s_i \in [0, 2]`
        defines how many times the radius must be considered as the step.
        A value of 2 means there is only one partition along the axis i, and
        a value of 1 means there are two partitions because the total
        considered length is twice the radius per axis. In general, a value of
        :math:`0 \leq s_i \leq 2` leads to
        :math:`\left\lceil{\dfrac{2}{s_i}}\right\rceil` partitions.
    :vartype cell_size: :class:`np.ndarray`
    :ivar dimensionality: The dimensionality of the space where the points are
        defined. Typically, it refers to the dimensionality of the structure
        space, i.e., the points without the features. For example, considering
        the :math:`(x, y)` coordinates of the points implies a dimensionality
        :math:`n=2`, while considering the :math:`(x, y, z)` coordinates of the
        points implies a dimensionality :math:`n=3`.
    :vartype dimensionality: int
    :ivar bounding_radii: The radii vector :math:`\pmb{r} = (r_1, \ldots, r_n)`
        that defines the radius for each axis. The length of the axis :math:`i`
        represented by the receptive field is given by
        :math:`2 \times r_i`, i.e., twice the axis radius.

        One simple way to compute the bounding radius for a set of coordinates
        :math:`X_i = \{x_{1i}, \ldots, x_{mi}\}` is:

        .. math::
            r_i = \dfrac{\max X_i - \min X_i}{2}
    :vartype bounding_radii: :class:`np.ndarray`
    :ivar num_cells: The number of cells :math:`R` composing the receptive
        field. It is computed as:

        .. math::
            R = \Biggl\lfloor\biggl({
                \prod_{i=1}^{n}{\dfrac{s_i}{2}}
            }\biggr)^{-1}\Biggr\rceil

        Where :math:`\lfloor x \rceil` means rounding to the nearest integer
        of x with round half toward positive infinity as tie-breaking rule.
    :vartype num_cells: int
    :ivar N: The indexing matrix :math:`\pmb{N}_{\geq -1}` described above. It
        is computed when calling
        :meth:`receptive_field_gs.ReceptiveFieldGS.fit`.
    :vartype N: :class:`np.ndarray`
    :ivar x: The center point of the receptive field. It is assigned when
        calling :meth:`receptive_field_gs.ReceptiveFieldGS.fit`.
    :vartype x: :class:`np.ndarray`
    :ivar m: The number of points represented in the receptive fields. It is
        assigned when calling :meth:`receptive_field_gs.ReceptiveFieldGS.fit`.
    :vartype m: int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a receptive field object.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveFieldGS.

        :Keyword Arguments:
            *   *bounding_radii* (``np.ndarray``) --
                The bounding radius for each axis defining the receptive field.
                See :class:`.ReceptiveFieldGS` for a more detailed description.
            *   *cell_size* (``np.ndarray``) --
                The cell size defining the receptive field. By default it is
                :math:`(0.05, 0.05, 0.05)`. See :class:`.ReceptiveFieldGS` for
                a more detailed description.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.cell_size = kwargs.get('cell_size', np.array([0.05, 0.05, 0.05]))
        self.dimensionality = self.cell_size.shape[0]
        self.bounding_radii = kwargs.get('bounding_radii', None)
        if self.bounding_radii is None:
            raise ValueError(
                'ReceptiveFieldGS must be instantiated for a given bounding '
                'radii. None were given.'
            )
        self.num_cells = ReceptiveFieldGS.num_cells_from_cell_size(
            self.cell_size
        )
        self.N = None  # The indexing matrix will be created during fit
        self.x = None  # The center point of the receptive field
        self.m = None  # Number of points the receptive field has been fit to

    # ---   RECEPTIVE FIELD METHODS   --- #
    # ----------------------------------- #
    def fit(self, X, x):
        r"""
        Fit the receptive field to represent the given points by building a
        grid containing them.

        The matrix of indices
        :math:`\pmb{N} \in \mathbb{Z}_{\geq -1}^{R \times R^*}` is computed
        here. See also
        :meth:`receptive_field_gs.ReceptiveFieldGS.shadow_indexing_matrix_from_points`
        .

        :param X: The input matrix of m points in an n-dimensional space.
        :type X: :class:`np.ndarray`
        :param x: The center point used to define the origin of the receptive
            field.
        :type x: :class:`np.ndarray`
        :return: The fit receptive field itself (for fluent programming).
        :rtype: :class:`.ReceptiveFieldGS`
        """
        # Validate input
        if x is None:
            raise ValueError(
                'ReceptiveFieldGS cannot fit without an input center point x.'
            )
        if X is None:
            raise ValueError(
                'ReceptiveFieldGS cannot fit without input points X.'
            )
        # Center and scale the input point cloud
        self.x = x
        X = self.center_and_scale(X)
        # Find the indexing matrix N
        self.N = self.shadow_indexing_matrix_from_points(X)
        # Store the number of points seen during fit
        self.m = X.shape[0]
        # Return self for fluent programming
        return self

    def centroids_from_points(self, X, interpolate=False, fill_centroid=False):
        r"""
        Compute the centroids, i.e., for each cell in the receptive field
        the point that represents the cell assuming the receptive field is
        applied to the rows of :math:`\pmb{X} \in \mathbb{R}^{m \times n}`
        understood as points.

        Let :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` be the matrix whose
        rows represent the centroids. Besides, consider the set of neighbors
        for a given point :math:`i` as
        :math:`\mathcal{N}_i = \lvert\left\{n_{ij} \geq 0 : 1 \leq j \leq R^* \rvert\right\}`
        , where :math:`n_{ij}` is the element at row :math:`i` column :math:`j`
        in the matrix :math:`\pmb{N}` (see :class:`.ReceptiveFieldGS`). For
        then, each centroid can be computed such that:

        .. math::
            \pmb{y}_{i*} =
                \lvert\mathcal{N}_{i}\rvert^{-1}
                \sum_{j \in \mathcal{N}_{i}}{\pmb{x}_{j*}}

        Regarding the interpolation, any missing centroid is defined
        considering the midrange point of the cell (i.e., the geometric center)
        and its :math:`3^{n}-1` closest neighbors. The coordinate-wise mean
        of these points yields the interpolated value. The reason why
        :math:`3^{n}-1` is selected as the number of neighbors for the
        interpolation is based on the idea that for 2D and 3D neighborhoods
        considering a cell in a grid and all its neighbor cells yields
        :math:`3^{n}` neighbors. One is discarded because it corresponds to the
        cell itself being interpolated. Consequently, :math:`3^{n}-1` is
        expected to be a reasonable compromise to avoid considering too much
        points so the interpolated point represents the entire receptive field
        instead of the local region where it belongs to. At the same time,
        the number of neighbors also scales with the dimensionality to provide
        an acceptable sample for a reliable interpolation.

        :param X: The matrix of input points
        :type X: :class:`np.ndarray`
        :param interpolate: True to interpolate missing centroids from
            non-missing centroids, False otherwise.
        :type interpolate: bool
        :param fill_centroid: When interpolate is False and fill_centroid is
            True, all missing points will be fill as the centroid of the
            input point cloud.
        :type fill_centroid: bool
        :return: A matrix which rows are the points representing the centroids.
        :rtype: :class:`np.ndarray`
        """
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
            missing_indices = np.flatnonzero(~not_nan_flags)
            sup_missing_Y = self.get_center_of_empty_cells(
                missing_indices=missing_indices
            )
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
        # Fill missing cells with centroid (only if no interpolation)
        elif fill_centroid:
            # Find centroid
            mu = np.mean(X, axis=0)
            # Assign centroid to missing indices
            Y[~not_nan_flags] = mu
        # Return
        return Y

    def propagate_values(self, v, safe=True):
        r"""
        Propagate given values, so they are associated to the points in
        :math:`\pmb{X}`.

        Propagating can be seen as a pseudo-injective map. Clearly, when
        :math:`R<m`, there is no injective map between :math:`\mathcal{R}` and
        :math:`\pmb{X}` because a single point in :math:`\mathcal{R}` can
        correspond to more than one point in :math:`\pmb{X}`. However,
        when propagating, each point in :math:`\mathcal{R}` can be associated
        to each corresponding point in :math:`\pmb{X}` thanks to the
        :math:`\pmb{N}` matrix.

        More concretely, propagating values can be seen as a map that
        transforms the given sequence of values :math:`v_1, \ldots, v_{R_p}`,
        where :math:`R_p \leq R` is the number of non-empty rows in
        :math:`\pmb{N}`. Note a row i is said to be empty when
        :math:`n_{ij} = -1, \forall 1 \leq j \leq R^*`. The output of the
        transformation is the sequence of values :math:`y_1, \ldots, y_m`
        such that :math:`y_i = v_j`, satisfying the point i in :math:`\pmb{X}`
        corresponds to the non-empty cell j in :math:`\mathcal{R}`.

        :param v: The values to be propagated. There must be one value per
            non-empty cell in the receptive field. The values must preserve
            the order of the non-empty cells. In other words, :math:`v_j` must
            either belong to the cell :math:`j` or the cell :math:`j+k` where
            `k` is the number of empty cells with index :math:`<j`.
        :type v: list
        :param safe: True to compute the propagation in safe mode, i.e.,
            raising an exception when there are NaN in the propagation. False
            otherwise.
        :type safe: bool
        :return: The output as a matrix when there are more than two values
            per point or the output as a vector when there is one value per
            point.
        :rtype: :class:`np.ndarray`
        """
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
                    'The GS receptive field propagated NaN values. This is '
                    'not allowed in safe mode.'
                )
        # Return output matrix (or vector if single-column)
        return Y if Y.shape[1] > 1 else Y.flatten()

    def reduce_values(self, X, v, reduce_f=np.mean, fill_nan=False):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{R \times n}` be the matrix of
        row-wise points such that the row i is the centroid that represents the
        cell i in the receptive field. Let :math:`\pmb{v} \in \mathbb{K}^{m}`
        be a vector that must be reduced to the receptive field to produce
        :math:`\pmb{y} \in \mathbb{K}^{R}`.

        For a given reduce function :math:`f` that maps an arbitrary number
        of input values to a single value, let :math:`u` represent its input.
        Note that :math:`u` represents any vector whose components are a subset
        of the set of components defining the vector :math:`v`. For then, the
        :meth:`reduce_values` procedure can be expressed as
        :math:`y_{i} = f(u_i)`.

        It might happen that some cells are empty and thus, no value is
        assigned to them during reduction. In those cases, it is possible to
        apply a filling rule that consists of assuming each empty cell has
        a value equal to that of its closest neighbor.

        :param X: The centroid representing each cell in matrix form. Each
            row i of X represents the centroid of a cell i from the receptive
            field.
        :type X: :class:`np.ndarray`
        :param v: The vector of values to reduce. The :math:`m` input
            components will be reduced to :math:`R` output components.
        :param reduce_f: The function to reduce many values to a single
            one. By default, it is mean.
        :type reduce_f: callable
        :param fill_nan: True to fill NaN values with the value of the closest
            non-empty cell in the receptive field.
        :type fill_nan: bool
        :return: The reduced vector.
        :rtype: :class:`np.ndarray`
        """
        # Reduce
        not_nan_flags = np.any(self.N >= 0, axis=1)
        v_reduced = np.array([
            reduce_f(v[Ni[Ni >= 0]]) if not_nan_flags[i] else np.nan
            for i, Ni in enumerate(self.N)
        ])
        # Fill NaN from closest neighbor (if requested)
        if fill_nan:
            # Prepare filling
            missing_indices = np.flatnonzero(~not_nan_flags)
            sup_missing_Y = self.get_center_of_empty_cells(
                missing_indices=missing_indices
            )
            # Find index of closest non-empty centroid
            non_empty_X = X[not_nan_flags]
            kdt = KDT(non_empty_X)
            I = kdt.query(sup_missing_Y, 1)[1]
            # Fill nan reduced values
            non_empty_v_reduced = v_reduced[not_nan_flags]
            v_reduced[missing_indices] = non_empty_v_reduced[I]
        # Return
        return v_reduced

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def shadow_indexing_matrix_from_points(self, X):
        r"""
        Compute the indexing matrix (supporting shadow indices)
        :math:`\pmb{N} \in \mathbb{Z}_{\geq -1}^{R \times R^*}` from the given
        points :math:`\pmb{X} \in \mathbb{R}^{m \times n}`.

        In this function, the points in :math:`\pmb{X}` are assumed to be
        expressed in the internal reference system of the receptive field.
        In other words, the points must have been translated to the origin
        and scaled so the considered interval for each axis is :math:`[-1, 1]`
        instead of the defined by the original units of the boundary radii.

        First, the dimensional factors :math:`f_1, \ldots, f_n` are computed
        such that :math:`f_1 = 1` and
        :math:`f_{i>1} = f_{i-1} \left\lceil\dfrac{2}{s_{i-1}}\right\rceil`.

        Then, the point-wise indices :math:`i_1, \ldots, i_m` can be computed
        such that:

        .. math::
            i_j = \sum_{k=1}^{n}{\min \biggl\{{
                f_k \biggl(
                    \biggl\lfloor{\dfrac{2}{s_k}}\biggr\rfloor - 1
                \biggr),
                f_k \biggl\lfloor{{
                    \dfrac{x_{jk}+1}{s_k}
                }\biggr\rfloor}
            }\biggr\}}

        The previous computation guarantees that extreme points, as the maximum
        vertex of the axis aligned bounding box containing the point cloud,
        are assigned to a proper cell. In other words, they are associated to
        an index inside the boundaries of the indexing space, i.e., no greater
        than the greatest supported index.

        Now, since the :math:`i_1, \ldots, i_m` indices are computed, it is
        known that the point :math:`j` must be represented in the :math:`i_j`
        row of matrix :math:`\pmb{N}`. Consequently, there must be one
        component of row :math:`\pmb{n}_{i_j}`, namely :math:`\pmb{n}_{i_jk}`,
        such that :math:`\pmb{n}_{i_jk} = j`. And thus, the matrix
        :math:`\pmb{N} \in \mathbb{Z}_{\geq -1}^{R \times R^*}` is built.

        :param X: The matrix representing the m input points to build the
            indexing matrix. The points must have been centered and scaled,
            so they are expressed in the internal reference system of the
            receptive field. See
            :meth:`receptive_field_gs.ReceptiveFieldGS.center_and_scale` for
            more information.
        :return: The built indexing matrix.
        :rtype: :class:`np.ndarray`
        """
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
            ((np.ceil(2/self.cell_size)-1) * dim_factors).astype(int)
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
        r"""
        Let :math:`\pmb{o}` be the origin of the receptive field (also referred
        to as center point), and :math:`\pmb{r}` be the bounding radii vector.

        Any row in :math:`\pmb{X}` is assumed to represent a point in the
        canonical reference system. Thus, it is possible to obtain a matrix
        :math:`\pmb{Y}` which rows are the points in :math:`\pmb{X}`
        transformed to the internal reference system of the receptive field.
        Each point in this matrix can be computed as:

        .. math::
            \pmb{y}_{i*} = \left[\begin{array}{ccc}
                \dfrac{x_{i1} - o_1}{r_1} &
                \cdots &
                \dfrac{x_{in} - o_n}{r_n}
            \end{array}\right]

        :param X: The matrix of points to be transformed.
        :type X: :class:`np.ndarray`
        :return: The matrix of transformed points.
        :rtype: :class:`np.ndarray`
        """
        return (X - self.x) / self.bounding_radii

    def undo_center_and_scale(self, X):
        r"""
        The inverse transform of the
        :meth:`receptive_field_gs.ReceptiveFieldGS.center_and_scale` method.

        .. math::
            \pmb{x}_{i*} = \left[\begin{array}{ccc}
                r_1 x_{i1} + o_1 &
                \cdots &
                r_n x_{in} + o_n
            \end{array}\right]

        :param X: The matrix of transformed points to be transformed back to
            their original representations.
        :type X: :class:`np.ndarray`
        :return: The matrix of points after reversing the transformation.
        :rtype: :class:`np.ndarray`
        """
        return self.bounding_radii * X + self.x

    def get_center_of_empty_cells(self, missing_indices=None):
        """
        Obtain the center point (computed as the midrange) for each empty cell.

        :param missing_indices: The indices of empty cells. It can be None,
            in that case it will be computed internally.
        :return: Matrix of geometric centers, one row per empty cell.
        :rtype: :class:`np.ndarray`
        """
        # Prepare
        b = np.ones(self.x.shape[0])  # Max vertex (1, 1, 1)
        a = -b  # Min vertex (-1, -1, -1)
        cells_per_axis = np.ceil((b-a)/self.cell_size).astype(int)
        if missing_indices is None:
            empty_flags = np.any(self.N < 0, axis=1)
            missing_indices = np.flatnonzero(~empty_flags)
        # Compute the steps per axis to define the centers
        num_steps = np.array([
            np.mod(
                np.floor(missing_indices / np.prod(cells_per_axis[:j])),
                cells_per_axis[j]
            ) for j in range(self.dimensionality)
        ]).T
        # Return centers of empty cells
        return a + num_steps * self.cell_size

    @staticmethod
    def num_cells_from_cell_size(cell_size):
        """
        Compute the number of cells for a receptive field defined by the given
        cell size.


        :param cell_size: The cell size for which the number of cells must be
            computed. See :class:`.ReceptiveFieldGS` for further details.
        :return: The number of cells composing the receptive field.
        :rtype: int
        """
        return int(np.prod(np.ceil(2/cell_size)))
