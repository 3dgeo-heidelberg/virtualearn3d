# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.utils.ptransf.receptive_field_hierarchical_fps import \
    ReceptiveFieldHierarchicalFPS  # TODO Rethink : Implement ReceptiveFieldHierarchicalFPS
import src.main.main_logger as LOGGING
import numpy as np
import joblib
import time

# ---   CLASS   --- #
# ----------------- #
class HierarchicalFPSPreProcecssor(ReceptiveFieldPreProcessor):
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to hierarchical neural networks such
    as hierarchical autoencoders.

    See :class:`.ReceptiveFieldHierarchicalFPS`.
    See :class:`.ReceptiveFieldPreProcessor`.

    :ivar num_downsampling_neighbors: List with the number of neighbors
        involved in the downsampling at a given depth, i.e., [i] corresponds
        to depth i.
    :vartype num_downsampling_neighbors: list
    :ivar num_pwise_neighbors: List with the number of k nearest neighbors
        for the point-wise feature extraction at a given depth, i.e., [i]
        gives the point-wise knn neighborhoods at depth i.
    :vartype num_pwise_neighbors: list
    :ivar num_upsampling_neighbors: List with the number of neighbors
        involved in the upsampling at a given depth, i.e., [i] corresponds
        to depth i.
    :vartype num_upsampling_neighbors: list
    :ivar num_points_per_depth: List with the number of points per depth
        level, i.e., the number of points per receptive field.
    :vartype num_points_per_depth: list
    :ivar depth: The depth of the hierarchical receptive fields. At building
        time it is taken as the length of the number of points per depth level.
    :vartype depth: int
    :ivar fast_flag_per_depth: List of boolean flags specifying whether
        the corresponding receptive field must be computed using a
        stochastic approximation (faster) or the exhaustive furthest point
        sampling computation (slower).
    :vartype fast_flag_per_depth: list
    """
    # ---   INIT  --- #
    # --------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Hierarchical Furthest Point
        Subsampling pre-processor.

        :param kwargs: The key-word arguments for the
            HierarchicalFurthestPointSubsamplingPreProcessor.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.num_downsampling_neighbors = kwargs.get(
            'num_downsampling_neighbors', None
        )
        self.num_pwise_neighbors = kwargs.get('num_pwise_neighbors', None)
        self.num_upsampling_neighbors = kwargs.get(
            'num_upsampling_neighbors', None
        )
        self.num_points_per_depth = kwargs.get('num_points_per_depth', None)
        self.depth = len(self.num_points_per_depth)
        self.fast_flag_per_depth = kwargs.get(
            'fast_flag_per_depth', [False for i in range(self.depth)]
        )
        # Validate attributes
        if(
            self.num_downsampling_neighbors is None or
            not isinstance(
                self.num_downsampling_neighbors, (list, tuple, np.ndarray)
            ) or
            len(self.num_downsampling_neighbors) < 1
        ):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor did not receive the '
                'depth-wise numbers of downsampling neighbors.'
            )
        if(
            self.num_upsampling_neighbors is None or
            not isinstance(
                self.num_upsampling_neighbors, (list, tuple, np.ndarray)
            ) or
            len(self.num_upsampling_neighbors) < 1
        ):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor did not receive the '
                'depth-wise numbers of upsampling neighbors.'
            )
        if(
            self.num_points_per_depth is None or
            not isinstance(
                self.num_points_per_depth, (list, tuple, np.ndarray)
            ) or
            len(self.num_points_per_depth) < 1
        ):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor did not receive '
                'a non-empty specification for the number of points per depth.'
            )
        if self.depth != len(self.fast_flag_per_depth):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor received '
                f'{len(self.fast_flag_per_depth)} fast flags but depth is '
                f'{self.depth} (they MUST be equal).'
            )
        if self.depth != len(self.num_downsampling_neighbors):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor received '
                f'{len(self.num_downsampling_neighbors)} downsampling '
                f'neighborhoods but depth is {self.depth} '
                '(they MUST be equal).'
            )
        if self.depth != len(self.num_pwise_neighbors):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor received '
                f'{len(self.num_pwise_neighbors)} point-wise neighborhoods but '
                f'depth is {self.depth} (they MUST be equal).'
            )
        if self.depth != len(self.num_upsampling_neighbors):
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor received '
                f'{len(self.num_upsampling_neighbors)} upsampling '
                f'neighborhoods but depth is {self.depth} '
                '(they MUST be equal).'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the pre-processing logic. It also updates the cache-like
        variables of the preprocessor.

        The pre-processing logic transforms the input structure space matrix
        :math:`\pmb{X_{\mathrm{IN}}} \in \mathbb{R}^{m_{\mathrm{IN}} \times n_x}`
        (typically :math:`n_x=3`), the feature space matrix
        :math:`\pmb{F_{\mathrm{IN}}} \in \mathbb{R}^{m_{\mathrm{IN}} \times n_f}`,
        and potentially the reference values
        :math:`\pmb{y} \in \mathbb^{R}{m_{\mathrm{IN}}}` into many receptive
        fields. Where :math:`m_{\mathrm{IN}}` is the number of input points.

        Now, a given receptive field can be represented by a structure space
        matrix :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}`, a feature space
        matrix :math:`\pmb{F} \in \mathbb{R}^{m \times n_f}, the downsampling
        indexing matrices
        :math:`\pmb{N}^{D}_{i} \in \mathbb{Z}^{m_i}`,
        the neighborhood indexing matrices
        :math:`\pmb{N}_{i} \in \mathbb{Z}^{m_i \times R}`,
        and the upsampling indexing matrices
        :math:`\pmb{N}^{U}_{i} \in \mathbb{Z}^{m_i}`,
        Where :math:`i = 1, \ldots, \text{max depth}`, :math:`m` is the
        number of points, :math:`m_i` is the number of points at depth i, and
        :math:`R in \mathbb{Z}_{>0}` the number of nearest neighbors defining
        each point-wise neighborhood.


        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a hierarchical model. If "X" is a list,
            then the first element is assumed to be the matrix
            :math:`\pmb{X_{\mathrm{IN}}}` of coordinates
            and the second the matrix
            :math:`\pmb{F_{\mathrm{IN}}}` of features. If "X" is a matrix
            (array), then the matrix of features is assumed to be a column
            vector of ones.
        :type inputs: dict
        :return: Either (X, F, ...NDi..., ...Ni..., ...NUi..., y) or
            (X, F, ...NDi..., ...Ni..., ...NUi...). Where X are the input
            points, F are the input features, NDi are point-wise downsamlping
            neighborhoods at depth i, Ni are the point-wise neighborhoods at
            depth i, and NUi are the point-wise upsampling neighborhoods at
            depth i.
        """
        # Extract inputs
        start = time.perf_counter()
        X, F, y = inputs['X'], None, inputs.get('y', None)
        if isinstance(X, list):
            X, F = X[0], X[1]
        else:
            F = np.ones((X.shape[0], 1), dtype=float)
        # Extract support neighborhoods
        sup_X, I = self.find_neighborhood(X, y=y)
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = HierarchicalFPSPreProcecssor.clean_support_neighborhoods(
            sup_X, I, self.num_points_per_depth[0]
        )
        # Export support points if requested
        # TODO Rethink : Implement
        self.last_call_neighborhoods = I
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveFieldHierarchicalFPS(
                num_points_per_depth=self.num_points_per_depth,
                num_encoding_neighbors=self.num_encoding_neighbors,
                fast_flag_per_depth=self.fast_flag_per_depth
            )
            for Ii in I
        ]
        # Extract downsampling neighborhoods
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
        if self.to_unit_sphere:
            Xout = np.array([
                ReceptiveFieldPreProcessor.transform_to_unit_sphere(
                    self.last_call_receptive_fields[i].centroids_from_points(
                        None
                    )
                )
                for i in range(len(I))
            ])
        else:
            Xout = np.array([
                self.last_call_receptive_fields[i].centroids_from_points(None)
                for i in range(len(I))
            ])
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The hierarchical FPS pre processor generated '
            f'{Xout.shape[0]} receptive fields with depth '
            f'{len(self.num_points_per_depth)} and the following points per '
            f'depth: {self.num_points_per_depth}'
        )
        # Features ready to be fed into the neural network
        rv = lambda rfi, F: [
            rfi.reduce_values(None, F[:, j]) for j in range(F.shape[1])
        ]
        Fout = np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(rv)(
                self.last_call_receptive_fields[i], F
            )
            for i in range(len(I))
        )).transpose([0, 2, 1])
        # Neighborhoods for hierarchical representation
        Dout = [
            self.last_call_receptive_fields[i].get_downsampling_matrices()
            for i in range(len(I))
        ]
        Nout = [
            self.last_call_receptive_fields[i].get_neighborhood_matrices()
            for i in range(len(I))
        ]
        Uout = [
            self.last_call_receptive_fields[i].get_upsampling_matrices()
            for i in range(len(I))
        ]
        # Handle labels
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The hierarchical FPS pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return [Xout, Fout] + Dout + Nout + Uout, yout
        LOGGING.LOGGER.info(
            'The hierarchical FPS pre processor pre-processed '
            f'{X.shape[0]} points for predictions in {end-start:.3f} seconds.'
        )
        # Return without labels
        return [Xout, Fout] + Dout + Nout + Uout
