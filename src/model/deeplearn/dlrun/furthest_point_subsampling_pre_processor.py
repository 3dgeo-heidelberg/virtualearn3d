# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import scipy.stats
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPreProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to some neural networks such as
    PointNet.

    See :class:`.ReceptiveFieldFPS`.

    :ivar num_points: The number of points any point cloud must be reduced to
        through furthest point subsampling.
    :vartype num_points: int
    :ivar num_encoding_neighbors: How many neighbors consider to propagate and
        also to reduce. See :class:`.ReceptiveFieldFPS` for further details.
    :vartype num_encoding_neighbors: int
    :ivar fast: Flag to control whether to use random methods to speed up the
        computation of the furthest point subsampling.
    :vartype fast: bool
    :ivar neighborhood_spec: The neighborhood specification. See the example
        below.

        .. code-block:: JSON

            {
                "type": "sphere",
                "radius": 5.0,
                "separation_factor": 1.0
            }

    :vartype neighborhood_spec: dict
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
        Initialization/instantiation of a Furthest Point Subsampling
        pre-processor.

        :param kwargs: The key-word arguments for the
            FurthestPointSubsamplingPreProcessor.
        """
        # Assign attributes
        self.num_points = kwargs.get('num_points', 8000)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 3)
        self.fast = kwargs.get('fast', False)
        self.neighborhood_spec = kwargs.get('neighborhood', None)
        if self.neighborhood_spec is None:
            raise DeepLearningException(
                'The FurthestPointSubsamplingPreProcessor did not receive '
                'any neighborhood specification.'
            )
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

        The pre-processing logic is similar to that of
        :meth:`grid_subsampling_pre_processor.GridSubsamplingPreProcessor.__call__`
        but using a :class:`.ReceptiveFieldFPS` instead of
        :class:`.ReceptiveFieldGS`.

        :param inputs: See
            :meth:`grid_subsampling_pre_processor.GridSubsamplingPrePRocessor.__call__`
            .
        :return: See
            :meth:`grid_subsampling_pre_processor.GridSubsamplingPrePRocessor.__call__`
            .
        """
        # Extract inputs
        start = time.perf_counter()
        X, y = inputs['X'], inputs.get('y', None)
        # Extract neighborhoods
        # TODO Rethink : Consider 2D support points for unbounded cylinders
        if self.neighborhood_spec['radius'] == 0:
            # The sphere of radius 0 is said to be the entire point cloud
            I = [np.arange(len(X), dtype=int).tolist()]
            sup_X = np.mean(X, axis=0).reshape(1, -1)
        else:
            # Spheres with a greater than zero radius
            sup_X = GridSubsamplingPreProcessor.build_support_points(
                X,
                self.neighborhood_spec['separation_factor'],
                self.neighborhood_spec['radius']
            )
            kdt = KDT(X)
            kdt_sup = KDT(sup_X)
            I = kdt_sup.query_ball_tree(kdt, self.neighborhood_spec['radius'])
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = FurthestPointSubsamplingPreProcessor\
            .clean_support_neighborhoods(
                sup_X, I, self.num_points
            )
        self.last_call_neighborhoods = I
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveFieldFPS(
                num_points=self.num_points,
                num_encoding_neighbors=self.num_encoding_neighbors,
                fast=self.fast
            )
            for Ii in I
        ]
        self.last_call_receptive_fields = joblib.Parallel(
            n_jobs=self.nthreads
        )(
            joblib.delayed(
                self.last_call_receptive_fields[i].fit
            )(
                X[Ii], sup_X[i]
            )
            for i, Ii in enumerate(I)
        )
        # Neighborhoods ready to be fed into the neural network
        Xout = np.array([
            self.last_call_receptive_fields[i].centroids_from_points(None)
            for i in range(len(I))
        ])
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor generated '
            f'{Xout.shape[0]} receptive fields.'
        )
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The furthest point subsampling pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return Xout, yout
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor pre-processed '
            f'{X.shape[0]} points for predictions in {end-start:.3f} seconds.'
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
        return self.num_points

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def clean_support_neighborhoods(sup_X, I, num_points):
        """
        Compute the clean version of the given support neighborhoods, i.e.,
        support points in sup_X and their neighborhoods as defined in I but
        considering only neighborhoods with more than num_points neighbors.

        :param sup_X: The matrix of coordinates representing the support
            points.
        :param I: The indices (in the original point domain) corresponding
            to each support point. In other words, I[i] gives the indices
            in X of support point i in X_sup.
        :return: The clean matrix of coordinates representing the support
            points and their neighborhoods.
        :rtype: tuple
        """
        # Remove neighborhoods with less than num_points neighbors
        non_empty_mask = [len(Ii) >= num_points for Ii in I]
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
                'FurthestPointSubsamplingPreProcessor cannot reduce labels '
                'because no neighborhood indices were given.'
            )
        # Compute and return the reduced labels
        return np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].reduce_values
            )(
                X_rf[i],
                y[Ii],
                reduce_f=lambda x: scipy.stats.mode(x)[0][0]
            ) for i, Ii in enumerate(I)
        ))
