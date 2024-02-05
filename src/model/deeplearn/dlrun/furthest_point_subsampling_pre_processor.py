# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.utils.neighborhood.support_neighborhoods import SupportNeighborhoods
import src.main.main_logger as LOGGING
import scipy.stats
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPreProcessor(ReceptiveFieldPreProcessor):
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to some neural networks such as
    PointNet.

    See :class:`.ReceptiveFieldFPS`.
    See :class:`.ReceptiveFieldPreProcessor`.

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
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.num_points = kwargs.get('num_points', 8000)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 3)
        self.fast = kwargs.get('fast', False)
        self.neighborhood_spec = kwargs.get('neighborhood', None)  # Support
        if self.neighborhood_spec is None:
            raise DeepLearningException(
                'The FurthestPointSubsamplingPreProcessor did not receive '
                'any neighborhood specification.'
            )

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
            :meth:`grid_subsampling_pre_processor.GridSubsamplingPreProcessor.__call__`
            .
        :return: See
            :meth:`grid_subsampling_pre_processor.GridSubsamplingPreProcessor.__call__`
            .
        """
        # Extract inputs
        start = time.perf_counter()
        X, F, y = inputs['X'], None, inputs.get('y', None)
        if isinstance(X, list):
            X, F = X[0], X[1]
        # Extract neighborhoods
        sup_X, I = self.find_neighborhood(X, y=y)
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = FurthestPointSubsamplingPreProcessor\
            .clean_support_neighborhoods(
                sup_X, I, self.num_points
            )
        self.last_call_neighborhoods = I
        # Export support points if requested
        self.export_support_points(inputs, sup_X)
        # Prepare receptive fields
        self.last_call_receptive_fields = [
            ReceptiveFieldFPS(
                num_points=self.num_points,
                num_encoding_neighbors=self.num_encoding_neighbors,
                fast=self.fast
            )
            for Ii in I
        ]
        self.fit_receptive_fields(X, sup_X, I)
        # Neighborhoods ready to be fed into the neural network
        Xout = self.handle_unit_sphere_transform(I)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor generated '
            f'{Xout.shape[0]} receptive fields of {self.num_points} points '
            'each.'
        )
        # Features ready to be fed into the neural network
        Fout = self.handle_features_reduction(
            F,
            len(I),  # number of neighborhoods
            lambda rfi, Xouti, F : [  # reduce function f(rf_i, Xout_i, F)
                rfi.reduce_values(None, F[:, j]) for j in range(F.shape[1])
            ]
        )
        # Handle labels
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The furthest point subsampling pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            if Fout is not None:
                return [Xout, Fout], yout
            else:
                return Xout, yout
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor pre-processed '
            f'{X.shape[0]} points for predictions in {end-start:.3f} seconds.'
        )
        # Return without labels
        if Fout is not None:
            return [Xout, Fout]
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
            to the receptive fields.
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
                reduce_f=lambda x: scipy.stats.mode(x)[0]
            ) for i, Ii in enumerate(I)
        ))

    def find_neighborhood(self, X, y=None):
        r"""
        Find the requested neighborhoods in the given input point cloud
        represented by the matrix of coordinates :math:`\pmb{X}`.

        :param X: The matrix of coordinates.
        :type X: :class:`np.ndarray`
        :param y: The vector of expected values (generally, class labels).
            It is an OPTIONAL argument that is only necessary when the
            neighborhoods must be found following a given class distribution.
        :type y: :class:`np.ndarray`
        :return: A tuple which first element are the support points
            representing the centers of the neighborhoods and which second
            element is a list of neighborhoods, where each neighborhood is
            represented by a list of indices corresponding to the rows (points)
            in :math:`\pmb{X}` that compose the neighborhood.
        :rtype: tuple
        """
        return SupportNeighborhoods(
            self.neighborhood_spec,
            support_strategy=self.support_strategy,
            support_strategy_num_points=self.support_strategy_num_points,
            support_strategy_fast=self.support_strategy_fast,
            support_chunk_size=self.support_chunk_size,
            training_class_distribution=self.training_class_distribution,
            center_on_pcloud=self.center_on_pcloud,
            nthreads=self.nthreads
        ).compute(X, y=y)

    # ---   SUPPORT POINTS EXPORT   --- #
    # --------------------------------- #
    def _export_support_points(self, sup_X, path):
        """
        See :class:`.ReceptiveFieldPreProcessor`,
        :meth:`receptive_field_pre_processor.ReceptiveFieldPreProcessor._export_support_points`,
        :class:`.GridSubsamplingPreProcessor`, and
        :meth:`grid_subsampling_pre_processor.GridSubsamplingPreProcessor.support_points_to_file`
        """
        return GridSubsamplingPreProcessor.support_points_to_file(sup_X, path)

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.overwrite_pretrained_model`
        method and
        :meth:`receptive_field_pre_processor.ReceptiveFieldPreProcessor.overwrite_pretrained_model`.
        """
        # Overwrite from parent
        super().overwrite_pretrained_model(spec)
        spec_keys = spec.keys()
        # Overwrite the attributes of the furth. pt. subsampling pre-processor
        if 'num_points' in spec_keys:
            self.num_points = spec['num_points']
        if 'num_encoding_neighbors' in spec_keys:
            self.num_encoding_neighbors = spec['num_encoding_neighbors']
        if 'fast' in spec_keys:
            self.fast = spec['fast']
        if 'neighborhood_spec' in spec_keys:
            self.neighborhood_spec = spec['neighborhood_spec']

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized furthest point
        subsampling pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__getstate__`.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Obtain parent's state
        state = super().__getstate__()
        # Update state
        state['num_points'] = self.num_points
        state['num_encoding_neighbors'] = self.num_encoding_neighbors
        state['fast'] = self.fast
        state['neighborhood_spec'] = self.neighborhood_spec
        # Return state
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized furthest point subsampling pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__setstate__`.

        :param state: The state's dictionary of the saved furthest point
            subsampling pre-processor.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state
        self.num_points = state['num_points']
        self.num_encoding_neighbors = state['num_encoding_neighbors']
        self.fast = state['fast']
        self.neighborhood_spec = state['neighborhood_spec']
