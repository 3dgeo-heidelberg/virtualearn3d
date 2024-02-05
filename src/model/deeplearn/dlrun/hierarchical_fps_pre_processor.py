# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor import \
    FurthestPointSubsamplingPreProcessor
from src.utils.ptransf.receptive_field_hierarchical_fps import \
    ReceptiveFieldHierarchicalFPS
from src.utils.neighborhood.support_neighborhoods import SupportNeighborhoods
import src.main.main_logger as LOGGING
import scipy
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class HierarchicalFPSPreProcessor(ReceptiveFieldPreProcessor):
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
        self.neighborhood_spec = kwargs.get('neighborhood', None)  # Support
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
        if self.neighborhood_spec is None:
            raise DeepLearningException(
                'The HierarchicalFPSPreProcessor did not receive any '
                'neighborhood specification.'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        r"""
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
        I, sup_X = HierarchicalFPSPreProcessor.clean_support_neighborhoods(
            sup_X, I, self.num_points_per_depth[0]
        )
        # Export support points if requested
        if inputs.get('plots_and_reports', True):
            if(
                inputs.get('training_support_points', False) and
                self.training_support_points_report_path is not None
            ):
                GridSubsamplingPreProcessor.support_points_to_file(
                    sup_X,
                    self.training_support_points_report_path
                )
            if(
                inputs.get('support_points', False) and
                self.support_points_report_path is not None
            ):
                GridSubsamplingPreProcessor.support_points_to_file(
                    sup_X,
                    self.support_points_report_path
                )
        self.last_call_neighborhoods = I
        # Prepare receptive fields
        self.last_call_receptive_fields = [
            ReceptiveFieldHierarchicalFPS(
                num_points_per_depth=self.num_points_per_depth,
                fast_flag_per_depth=self.fast_flag_per_depth,
                num_downsampling_neighbors=self.num_downsampling_neighbors,
                num_pwise_neighbors=self.num_pwise_neighbors,
                num_upsampling_neighbors=self.num_upsampling_neighbors
            )
            for Ii in I
        ]
        self.fit_receptive_fields(X, sup_X, I)
        # Neighborhoods ready to be fed into the neural network
        Xout = self.handle_unit_sphere_transform(I)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The hierarchical FPS pre processor generated '
            f'{Xout.shape[0]} receptive fields with depth '
            f'{len(self.num_points_per_depth)} and the following points per '
            f'depth: {self.num_points_per_depth}'
        )
        # Features ready to be fed into the neural network
        Fout = self.handle_features_reduction(
            F,
            len(I),  # number of neighborhoods
            lambda rfi, Xouti, F: [  # reduce function f(rf_i, Xout_i, F)
                rfi.reduce_values(None, F[:, j]) for j in range(F.shape[1])
            ]
        )
        if Fout is None:
            raise DeepLearningException(
                'HierarchicalFPSPreProcessor yielded a null Fout (None). '
                'This MUST not happen.'
            )
        # Structure spaces strictly AFTER depth 1
        Xdout = [
            np.array([
                self.last_call_receptive_fields[i].Ys[d]
                for i in range(len(I))
            ])
            for d in range(1, self.depth)
        ]
        if self.to_unit_sphere:
            Xdout = list(map(
                ReceptiveFieldPreProcessor.transform_to_unit_sphere,
                Xdout
            ))
        # Neighborhoods for hierarchical representation
        Dout = np.array([
            self.last_call_receptive_fields[i].get_downsampling_matrices()
            for i in range(len(I))
        ], dtype='object').T.tolist()
        Nout = np.array([
            self.last_call_receptive_fields[i].get_neighborhood_matrices()
            for i in range(len(I))
        ], dtype='object').T.tolist()
        Uout = np.array([
            self.last_call_receptive_fields[i].get_upsampling_matrices()
            for i in range(len(I))
        ], dtype='object').T.tolist()
        # Prepare basic output
        out = [Xout, Fout] + Xdout + Dout[1:] + Nout + Uout[1:]
        out = [np.array(X) for X in out]
        # Handle labels
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The hierarchical FPS pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return out, yout
        LOGGING.LOGGER.info(
            'The hierarchical FPS pre processor pre-processed '
            f'{X.shape[0]} points for predictions in {end-start:.3f} seconds.'
        )
        # Return with no labels
        return out

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def clean_support_neighborhoods(sup_X, I, num_points):
        """
        See :class:`.FurthestPointSubsamplingPreProcessor` and
        :meth:`furthest_point_subsampling_pre_processor.FurthestPointSubsamplingPreProcessor.clean_support_neighborhoods`.
        """
        return FurthestPointSubsamplingPreProcessor.clean_support_neighborhoods(
            sup_X, I, num_points
        )

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
                'HierarchicalFPSPreProcessor cannot reduce labels '
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
        """
        See :class:`.FurthestPointSubsamplingPreProcessor` and
        :meth:`furthest_point_subsampling_pre_processor.FurthestPointSubsamplingPreProcessor.find_neighborhood`.
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

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        See
        :meth:`hierarchical_pre_processor.HierarchicalPreProcessor.overwrite_pretrained_model`
        method and
        :meth:`receptive_field_pre_processor.ReceptiveFieldPreProcessor.overwrite_pretrained_model`.
        """
        # Overwrite from parent
        super().overwrite_pretrained_model(spec)
        spec_keys = spec.keys()
        # Overwrite the attributes of the hierarchical FPS pre-processor
        if 'num_downsampling_neighbors' in spec_keys:
            self.num_downsampling_neighbors = spec['num_downsampling_neighbors']
        if 'num_pwise_neighbors' in spec_keys:
            self.num_pwise_neighbors = spec['num_pwise_neighbors']
        if 'num_upsampling_neighbors' in spec_keys:
            self.num_upsampling_neighbors = spec['num_upsampling_neighbors']
        if 'num_points_per_depth' in spec_keys:
            self.num_points_per_depth = spec['num_points_per_depth']
        if 'depth' in spec_keys:
            self.depth = spec['depth']
        if 'fast_flag_per_depth' in spec_keys:
            self.fast_flag_per_depth = spec['fast_flag_per_depth']
        if 'neighborhood_spec' in spec_keys:
            self.neighborhood_spec = spec['neighborhood_spec']

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized hierarchical furthest
        point sampling receptive field pre-processor.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Obtain parent's state
        state = super().__getstate__()
        # Update state
        state['num_downsampling_neighbors'] = self.num_downsampling_neighbors
        state['num_pwise_neighbors'] = self.num_pwise_neighbors
        state['num_upsampling_neighbors'] = self.num_upsampling_neighbors
        state['num_points_per_depth'] = self.num_points_per_depth
        state['depth'] = self.depth
        state['fast_flag_per_depth'] = self.fast_flag_per_depth
        state['neighborhood_spec'] = self.neighborhood_spec
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized hierarchical furthest point subsampling pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__setstate__`.

        :param state: The state's dictionary of the saved hierarchical
            furthest point subsampling pre-processor.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state
        self.num_downsampling_neighbors = state['num_downsampling_neighbors']
        self.num_pwise_neighbors = state['num_pwise_neighbors']
        self.num_upsampling_neighbors = state['num_upsampling_neighbors']
        self.num_points_per_depth = state['num_points_per_depth']
        self.depth = state['depth']
        self.fast_flag_per_depth = state['fast_flag_per_depth']
        self.neighborhood_spec = state['neighborhood_spec']
