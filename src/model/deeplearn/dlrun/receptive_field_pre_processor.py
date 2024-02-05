# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import joblib
import numpy as np
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldPreProcessor:
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to some neural networks such as
    PointNet.

    More concretely, this abstract class implements only the common logic for
    any pre-processor based on receptive fields. For example, grid
    subsampling pre-processor, or furthest point subsampling pre-processor.

    See :class:`.ReceptiveField`.
    See :class:`.GridSubsamplingPreProcessor`
    See :class:`.FurthestPointSubsamplingPreProcessor`

    :ivar support_strategy: The strategy to be used to compute the support
        points when no training class distribution has been given. It can be
        "grid" (default) to get support points through grid sampling, or "fps"
        to use furthest point sampling.
    :vartype support_strategy: str
    :ivar support_strategy_num_points: The number of points to consider when
        using a furthest point sampling strategy to compute the support points.
    :vartype support_strategy_num_points: int
    :ivar support_chunk_size: The number of supports per chunk :math:`n`.
        If zero, all support points are considered at once.
        If greater than zero, the support points will be considered in chunks
        of :math:`n` points.
    :vartype support_chunk_size: int
    :ivar to_unit_sphere: Whether to map the structure space of the receptive
        fields to the unit sphere (True) or not (False).
    :vartype to_unit_sphere: bool
    :ivar training_class_distribution: The target class distribution to
        consider when building the support points. The element i of this
        vector (or list) specifies how many neighborhoods centered on a point
        of class i must be considered. If None, then the point-wise classes
        will not be considered when building the neighborhoods.
    :vartype training_class_distribution: list or tuple or :class:`np.ndarray`
    :ivar center_on_pcloud: When True, the support points defining the
        receptive field will be transformed to be a point from the input
        point cloud. Consequently, the generated neighborhoods are centered on
        points that belong to the point cloud instead of arbitrary support
        points. When False, support points do not necessarily match points
        from the point cloud.
    :vartype center_on_pcloud: bool
    :ivar receptive_fields_dir: Directory where the point clouds representing
        the many receptive fields will be exported (OPTIONAL).
    :vartype receptive_fields_dir: str or None
    :ivar receptive_fields_distribution_report_path: Path where the text-like
        report on the class distribution of the receptive fields will be
        exported (OPTIONAL).
    :vartype receptive_fields_distribution_report_path: str or None
    :ivar receptive_fields_distribution_plot_path: Path where the plot
        representing the class distribution of the receptive fields will be
        exported (OPTIONAL).
    :vartype receptive_fields_distribution_plot_path: str or None
    :ivar training_receptive_fields_dir: Like receptive_fields_dir but for
        the receptive fields at training.
    :ivar training_receptive_fields_distribution_report_path: Like
        receptive_fields_distribution_report_path but for the receptive fields
        at training.
    :ivar training_receptive_fields_distribution_plot_path: Like
        receptive_fields_distribution_plot_path but for the receptive fields
        at training.
    :ivar last_call_receptive_fields: List of the receptive fields used the
        last time that the pre-processing logic was executed.
    :vartype last_call_receptive_fields: list
    :ivar last_call_neighborhoods: List of neighborhoods (represented by
        indices) used the last time that the pre-processing logic was
        executed.
    :vartype last_call_neighborhoods: list
    """
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Receptive Field pre-processor.

        :param kwargs: The key-word arguments for the
            ReceptiveFieldPreProcessor.
        """
        # Assign attributes
        self.support_strategy = kwargs.get('support_strategy', 'grid')
        self.support_strategy_num_points = kwargs.get(
            'support_strategy_num_points', 1000
        )
        self.support_strategy_fast = kwargs.get('support_strategy_fast', False)
        self.support_chunk_size = kwargs.get('support_chunk_size', 0)
        self.to_unit_sphere = kwargs.get('to_unit_sphere', False)
        self.training_class_distribution = kwargs.get(
            'training_class_distribution', None
        )
        self.center_on_pcloud = kwargs.get('center_on_pcloud', False)
        self.nthreads = kwargs.get('nthreads', 1)
        self.receptive_fields_distribution_report_path = kwargs.get(
            'receptive_fields_distribution_report_path', None
        )
        self.receptive_fields_distribution_plot_path = kwargs.get(
            'receptive_fields_distribution_plot_path', None
        )
        self.receptive_fields_dir = kwargs.get('receptive_fields_dir', None)
        self.training_receptive_fields_dir = kwargs.get(
            'training_receptive_fields_dir', None
        )
        self.training_receptive_fields_distribution_report_path = kwargs.get(
            'training_receptive_fields_distribution_report_path', None
        )
        self.training_receptive_fields_distribution_plot_path = kwargs.get(
            'training_receptive_fields_distribution_plot_path', None
        )
        self.training_support_points_report_path = kwargs.get(
            'training_support_points_report_path', None
        )
        self.support_points_report_path = kwargs.get(
            'support_points_report_path', None
        )
        # Initialize last call cache
        self.last_call_receptive_fields = None
        self.last_call_neighborhoods = None

    # ---   RUN / CALL   --- #
    # ---------------------- #
    @abstractmethod
    def __call__(self, inputs):
        """
        Any receptive field pre-processor must override this abstract method to
        implement the pre-processing logic.
        """
        pass

    # ---   SUPPORT POINTS EXPORT   --- #
    # --------------------------------- #
    def export_support_points(self, inputs, sup_X):
        """
        Handle the logic to export the support points.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train the model. If "X" is a list, then the
            first element is assumed to be the matrix X of coordinates and
            the second the matrix F of features.
        :type inputs: dict
        :param sup_X: The structure space matrix representing the support
            points.
        :type sup_X: :class:`np.ndarray`
        :return: Nothing at all, but output files might be written.
        """
        # Check whether exporting support points is required
        if not inputs.get('plots_and_reports', True):
            return
        # Check if particularly training support points must be exported
        if(
            inputs.get('training_support_points', False) and
            self.training_support_points_report_path is not None
        ):
            self._export_support_points(
                sup_X, self.training_support_points_report_path
            )
        # Check if particularly non-trainign support points must be exported
        if(
            inputs.get('support_points', False) and
            self.support_points_report_path is not None
        ):
            self._export_support_points(
                sup_X,
                self.support_points_report_path
            )

    def _export_support_points(self, sup_X, path):
        """
        Any receptive field pre-processor that needs to export support points
        must override this method to implement the corresponding logic.

        :param sup_X: The support points to be exported
        :type sup_X: :class:`np.ndarray`
        :param path: The path where the support points must be written.
        :type path: str
        :return: Nothing at all, but an output file is generated.
        """
        raise DeepLearningException(
            'ReceptiveFieldPreProcessor._export_support_points must not '
            'be called. Any receptive field pre-processor that must handle '
            'the exporting of support points needs to override the '
            '_export_support_points method to provide a valid logic.'
        )

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.overwrite_pretrained_model`
        method.
        """
        spec_keys = spec.keys()
        # Overwrite the attributes of the furth. pt. subsampling pre-processor
        if 'support_strategy' in spec_keys:
            self.support_strategy = spec['support_strategy']
        if 'support_strategy_num_points' in spec_keys:
            self.support_strategy_num_points = spec[
                'support_strategy_num_points'
            ]
        if 'support_chunk_size' in spec_keys:
            self.support_chunk_size = spec['support_chunk_size']
        if 'training_class_distribution' in spec_keys:
            self.training_class_distribution = spec[
                'training_class_distribution'
            ]
        if 'center_on_pcloud' in spec_keys:
            self.center_on_pcloud = spec['center_on_pcloud']
        if 'nthreads' in spec_keys:
            self.nthreads = spec['nthreads']
        if 'receptive_fields_dir' in spec_keys:
            self.receptive_fields_dir = spec['receptive_fields_dir']
        if 'receptive_fields_distribution_report_path' in spec_keys:
            self.receptive_fields_distribution_report_path = spec[
                'receptive_fields_distribution_report_path'
            ]
        if 'receptive_fields_distribution_plot_path' in spec_keys:
            self.receptive_fields_distribution_plot_path = spec[
                'receptive_fields_distribution_plot_path'
            ]
        if 'training_receptive_fields_dir' in spec_keys:
            self.training_receptive_fields_dir = spec[
                'training_receptive_fields_dir'
            ]
        if 'training_receptive_fields_distribution_report_path' in spec_keys:
            self.training_receptive_fields_distribution_report_path = spec[
                'training_receptive_fields_distribution_report_path'
            ]
        if 'training_receptive_fields_distribution_plot_path' in spec_keys:
            self.training_receptive_fields_distribution_plot_path = spec[
                'training_receptive_fields_distribution_plot_path'
            ]
        if 'training_support_points_report_path' in spec_keys:
            self.training_support_points_report_path = spec[
                'training_support_points_report_path'
            ]
        if 'support_points_report_path' in spec_keys:
            self.support_points_report_path = spec[
                'support_points_report_path'
            ]

    def update_paths(self, preproc):
        """
        Consider the given specification of pre-processing arguments to update
        the paths.
        """
        # Nothing to do if no specification is given
        if preproc is None:
            return
        # Update paths
        self.training_receptive_fields_distribution_report_path = \
            preproc['training_receptive_fields_distribution_report_path']
        self.training_receptive_fields_distribution_plot_path = \
            preproc['training_receptive_fields_distribution_plot_path']
        self.training_receptive_fields_dir = \
            preproc['training_receptive_fields_dir']
        self.receptive_fields_distribution_report_path = \
            preproc['receptive_fields_distribution_report_path']
        self.receptive_fields_distribution_plot_path = \
            preproc['receptive_fields_distribution_plot_path']
        self.receptive_fields_dir = \
            preproc['receptive_fields_dir']
        self.training_support_points_report_path = \
            preproc['training_support_points_report_path']
        self.support_points_report_path = \
            preproc['support_points_report_path']

    @staticmethod
    def transform_to_unit_sphere(X):
        r"""
        Map the points in :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` to the
        unit sphere (typically :math:`n_x = 3`, i.e., 3D).

        Let :math:`r = \max_{1 \leq i \leq m} \; \rVert{\pmb{x}_{i*}}\rVert^2`
        where :math:`\pmb{x}_{i*}` represents the i-th row (i.e., point) in
        the matrix :math:`\pmb{X}`. For then, the structure space transformed
        to the unit sphere can be represented as follows:

        .. math::

            \pmb{X'} = \pmb{X}/r

        :param X: The matrix representing a structure space such that rows
            are points and columns are coordinates.
        :type X: :class:`np.ndarray`
        :return: The structure space matrix transformed to the unit sphere.
        :rtype: :class:`np.ndarray`
        """
        squared_distances = np.sum(np.power(X, 2), axis=1)
        r = np.sqrt(np.max(squared_distances))
        return X/r

    def handle_unit_sphere_transform(self, I, **kwargs):
        """
        Handle the transformation to the unit sphere of the point-wise
        coordinates for each input structure space, if requested.

        Note that this method might be overriden or ignored by pre-processors
        providing an alternative strategy to handle the transform of the
        coordinates to the unit sphere.

        :param I: The list of neighborhoods such that I[i] is the i-th
            neighborhood, represented by the indices of the points that belong
            to that neighborhood. Typically, neighborhoods are centered on the
            support points.
        :type I: list of list
        :param kwargs: Any extra argument given as a key-value pair.
        :type kwargs: dict
        :return: The structure space matrices ready to be fed into the neural
            network.
        :rtype: :class:`np.ndarray`
        """
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
        return Xout

    def fit_receptive_fields(self, X, sup_X, I):
        """
        Update the self.last_call_receptive_fields attribute by calling the
        fit method of each receptive field. The fitted receptive fields
        replace the previously available receptive fields.

        :param X: The structure space matrix representing the point cloud.
        :type X: :class:`np.ndarray`
        :param sup_X: The structure space matrix representing the support
            points defining the receptive fields (typically the centers).
        :type sup_X: :class:`np.ndarray`
        :param I: The list of neighborhoods such that I[i] is the i-th
            neighborhood, represented by the indices of the points that belong
            to that neighborhood. Typically, neighborhoods are centered on the
            support points.
        :type I: list of list
        :return: The self.fit_receptive_fields attribute after the update.
        :rtype: list
        """
        self.last_call_receptive_fields = joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].fit
            )(
                X[Ii], sup_X[i]
            )
            for i, Ii in enumerate(I)
        )
        return self.last_call_receptive_fields

    def handle_features_reduction(self, F, num_neighborhoods, rv, Xout=None):
        """
        Handle the features reduction operation when the receptive field is
        called. In doing so, a reduce value function (rv) from the
        corresponding type of receptive field is often used.

        See :class:`.ReceptiveField` and
        :meth:`receptive_field.ReceptiveField.reduce_values`.

        :param F: The matrix of features.
        :param num_neighborhoods: The number of neighborhoods involved in the
            reduction.
        :param rv: The reduce value function that receives three arguments as
            input. The i-th receptive field (rf_i), the structure space matrix
            for the i-th receptive field (Xout_i), and the matrix of features
            (F).
        :type rv: Callable
        :param Xout: The structure space matrix for each receptive field. It
            can be None (default). In this case, it will be considered that
            each receptive field has a None structure space matrix and that it
            will be handled by the logic of the receptive field.
        :type Xout: list of :class:`np.ndarray` or :class:`np.ndarray` as a
            tensor whose slices represent receptive fields.
        :return: The reduced matrix of features for each receptive field.
        :rtype: :class:`np.ndarray` as a tensor whose slices represent
            receptive fields.
        """
        # Check whether there are features to reduce
        if F is None or len(F) <= 0:
            return None
        # Handle Xout
        if Xout is None:
            Xout = [None for i in range(num_neighborhoods)]
        # Reduce features
        return np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(rv)(
                self.last_call_receptive_fields[i], Xout[i], F
            )
            for i in range(num_neighborhoods)
        )).transpose([0, 2, 1])

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized receptive field
        pre-processor.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Return pre-processor state (cache to None)
        return {
            'support_strategy': self.support_strategy,
            'support_strategy_num_points': self.support_strategy_num_points,
            'support_strategy_fast': self.support_strategy_fast,
            'support_chunk_size': self.support_chunk_size,
            'to_unit_sphere': self.to_unit_sphere,
            'training_class_distribution': self.training_class_distribution,
            'center_on_pcloud': self.center_on_pcloud,
            'nthreads': self.nthreads,
            'receptive_fields_dir': None,
            'receptive_fields_distribution_report_path': None,
            'receptive_fields_distribution_plot_path': None,
            'training_receptive_fields_dir': None,
            'training_receptive_fields_distribution_report_path': None,
            'training_receptive_fields_distribution_plot_path': None,
            'training_support_points_report_path': None,
            'support_points_report_path': None,
            # Cache attributes below
            'last_call_receptive_fields': None,
            'last_call_neighborhoods': None
        }

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized receptive field pre-processor.

        :param state: The state's dictionary of the saved receptive field
            pre-processor.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign member attributes from state
        self.support_strategy = state.get('support_strategy', 'grid')
        self.support_strategy_num_points = state.get(
            'support_strategy_num_points', 1000
        )
        self.support_strategy_fast = state.get('support_strategy_fast', False)
        self.support_chunk_size = state.get('support_chunk_size', 0)
        self.to_unit_sphere = state.get('to_unit_sphere', False)
        self.training_class_distribution = state['training_class_distribution']
        self.center_on_pcloud = state['center_on_pcloud']
        self.nthreads = state['nthreads']
        self.receptive_fields_dir = None
        self.receptive_fields_distribution_report_path = None
        self.receptive_fields_distribution_plot_path = None
        self.training_receptive_fields_dir = None
        self.training_receptive_fields_distribution_report_path = None
        self.training_receptive_fields_distribution_plot_path = None
        self.training_support_points_report_path = None
        self.support_points_report_path = None
        self.last_call_receptive_fields = None
        self.last_call_neighborhoods = None
