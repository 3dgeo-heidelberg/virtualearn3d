# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_gs import ReceptiveFieldGS
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import scipy.stats
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class GridSubsamplingPreProcessor:
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to some neural networks such as
    PointNet.

    See :class:`.ReceptiveFieldGS`.

    :ivar sphere_radius: The radius of the sphere that bounds a neighborhood.
        For an arbitrary set of 3D points it can be found as follows:

        .. math::
            r = \dfrac{1}{2} \, \max \; \biggl\{
                x_{\mathrm{max}}-x_{\mathrm{min}},
                y_{\mathrm{max}}-y_{\mathrm{min}},
                z_{\mathrm{max}}-z_{\mathrm{min}}
            \biggr\}

    :vartype sphere_radius: float
    :ivar separation_factor: How many times the sphere radius separates the
        support points to find the input neighborhoods that will be used to
        generate the receptive fields for the neural network.

        For a given separation factor :math:`k`, the following condition
        should be satisfied to prevent missing any region of the input
        point cloud on a :math:`n`-dimensional space:

        .. math::
            k \leq \dfrac{2}{\sqrt{n}}

    :vartype separation_factor: float
    :ivar cell_size: The cell size defining the receptive field. See
        :class:`.ReceptiveFieldGS`.
    :vartype cell_size: :class:`np.ndarray`
    :ivar receptive_fields_dir: Directory where the point clouds representing
        the many receptive fields will be exported (OPTIONAL). If given, it
        will be used by the post-processor.
    :vartype receptive_fields_dir: str or None
    :ivar last_call_receptive_fields: List of the receptive fields used the
        last time that the pre-processing logic was executed.
    :vartype last_call_receptive_fields: list
    :ivar last_call_neighborhoods: List of neighborhoods (represented by
        indices) used the last time that the pre-processing logic was
        executed.
    :vartype last_call_neighborhoods: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Grid Subsampling pre-processor.

        :param kwargs: The key-word arguments for the
            GridSubSamplingPreProcessor.
        """
        # Assign attributes
        self.sphere_radius = kwargs.get('sphere_radius', 1.0)
        self.separation_factor = kwargs.get('separation_factor', np.sqrt(3)/4)
        self.cell_size = np.array(kwargs.get('cell_size', [0.1, 0.1, 0.1]))
        self.interpolate = kwargs.get('interpolate', True)
        self.nthreads = kwargs.get('nthreads', 1)
        self.receptive_fields_dir = kwargs.get('receptive_fields_dir', None)
        # Initialize last call cache
        self.last_call_receptive_fields = None
        self.last_call_neighborhoods = None

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the pre-processing logic. It also updates the cache-like
        variables of the preprocessor.

        The pre-processing logic consists of steps:

        **1) Generate support points.**

        Support points are generated as a set of points separated in :math:`kr`
        units between them and covering the entire point cloud. Watch out,
        generating support points with
        :math:`k > 2/\sqrt{n}` in a :math:`n`-dimensional
        space leads to gaps between the bounding spheres centered on the
        support points.

        **2) Find neighborhoods centered on support points.**
        For each support point, a neighborhood (by default a spherical
        neighborhood but the neighborhood definition can be arbitrarily
        changed) centered on it is found.

        **3) Filter empty neighborhoods.**
        Any support point that leads to an empty neighborhood when considering
        as neighbors points from the input point cloud is filtered out.

        **4) Transform non-empty neighborhoods to receptive fields.**
        Each non-empty neighborhood is transformed to a receptive field.
        See :class:`.ReceptiveFieldGS` for further details.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a PointNet model.
        :type inputs: dict
        :return: Either (Xout, yout) or Xout. Where Xout are the points
            representing the receptive field and yout (only given when
            "y" was given in the inputs dictionary) the corresponding
            reference values for those points.
        """
        # Extract inputs
        start = time.perf_counter()
        X, y = inputs['X'], inputs.get('y', None)
        # Extract neighborhoods
        # TODO Rethink : Chunk queries to prevent OOM ?
        sup_X = GridSubsamplingPreProcessor.build_support_points(
            X,
            self.separation_factor,
            self.sphere_radius
        )
        kdt = KDT(X)
        kdt_sup = KDT(sup_X)
        I = kdt_sup.query_ball_tree(kdt, self.sphere_radius)  # Neigh. indices
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = GridSubsamplingPreProcessor.clean_support_neighborhoods(
            sup_X, I
        )
        self.last_call_neighborhoods = I
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveFieldGS(
                bounding_radii=np.array([
                    self.sphere_radius for j in range(X.shape[1])
                ]),
                cell_size=self.cell_size
            )
            for Ii in I
        ]
        self.last_call_receptive_fields = joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].fit
            )(
                X[Ii], sup_X[i]
            )
            for i, Ii in enumerate(I)
        )
        # Neighborhoods ready to be fed into the neural network
        # TODO Rethink : Use support points to build the input ?
        # TODO Rethink : Maybe class-based picking for training ?
        Xout = np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].centroids_from_points
            )(
                X[Ii],
                interpolate=self.interpolate,
                fill_centroid=not self.interpolate
            )
            for i, Ii in enumerate(I)
        ))
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The grid subsampling pre processor generated {Xout.shape[0]} '
            'receptive fields. '
        )
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The grid subsampling pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return Xout, yout
        LOGGING.LOGGER.info(
            f'The grid subsampling pre processor pre-processed {X.shape[0]} '
            f'points for predictions in {end-start:.3f} seconds.'
        )
        return Xout

    # ---   POINT-NET METHODS   --- #
    # ----------------------------- #
    def get_num_input_points(self):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.get_num_input_points`
        .
        """
        return ReceptiveFieldGS.num_cells_from_cell_size(self.cell_size)

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def build_support_points(X, separation_factor, sphere_radius):
        r"""
        Compute the support points separated :math:`k` times the radius
        :math:`r` distributed along the bounding box defining the boundaries
        of :math:`\pmb{X}`.

        :param X: `\pmb{X}`, i.e., the matrix of coordinates representing the
            input points.
        :param separation_factor: :math:`k`
        :param sphere_radius: :math:`r`
        :return: The support points as a matrix where rows are support points
            and columns are coordinates.
        :rtype: :class:`np.ndarray`
        """
        xmin, xmax = np.min(X, axis=0), np.max(X, axis=0)
        l = separation_factor * sphere_radius  # Cell size
        A, B, C = np.meshgrid(
            np.concatenate([np.arange(xmin[0], xmax[0], l), [xmax[0]]]),
            np.concatenate([np.arange(xmin[1], xmax[1], l), [xmax[1]]]),
            np.concatenate([np.arange(xmin[2], xmax[2], l), [xmax[2]]])
        )
        return np.array([A.flatten(), B.flatten(), C.flatten()]).T

    @staticmethod
    def clean_support_neighborhoods(sup_X, I):
        """
        Compute the clean version of the given support neighborhoods, i.e.,
        support points in sup_X and their neighborhoods as defined in I but
        considering only non-empty neighborhoods.

        :param sup_X: The matrix of coordinates representing the support
            points.
        :param I: The indices (in the original point domain) corresponding
            to each support point. In other words, I[i] gives the indices
            in X of the support point i in X_sup.
        :return: The clean matrix of coordinates representing the support
            points and their neighborhoods.
        :rtype: tuple
        """
        non_empty_mask = [len(Ii) > 0 for Ii in I]
        I = [Ii for i, Ii in enumerate(I) if non_empty_mask[i]]
        sup_X = sup_X[non_empty_mask]
        return I, sup_X

    def reduce_labels(self, X_rf, y, I=None):
        r"""
        Reduce the given labels :math:`\pmb{y} \in \mathbb{Z}_{\geq 0}^{m}`
        to the receptive field labels
        :math:`\pmb{y}_{\mathrm{rf}} \in \mathbb{Z}_{\geq 0}^{R}`.

        :param X_rf: The matrices of coordinates representing the receptive
            fields.
        :type X_rf: :class:`np.ndarray`
        :param y: The labels of the original point cloud that must be reduced
            to the receptive field.
        :type y: :class:`np.ndarray`
        :param I: The list of neighborhoods. Each element of I is itself a list
            of indices that represents the neighborhood in the point cloud
            that corresponds to the point in the receptive field.
        :type I: list
        :return: The reduced labels for each receptive field.
        """

        # Handle automatic neighborhoods from cache
        if I is None:
            I = self.last_call_neighborhoods
        # Validate neighborhoods are given
        if I is None or len(I) < 1:
            raise DeepLearningException(
                'GridSubsamplingPreProcessor cannot reduce labels because '
                'no neighborhood indices were given.'
            )
        # Compute and return the reduced labels
        return np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].reduce_values
            )(
                X_rf[i],
                y[Ii],
                reduce_f=lambda x: scipy.stats.mode(x)[0][0],
                fill_nan=True
            ) for i, Ii in enumerate(I)
        ))
