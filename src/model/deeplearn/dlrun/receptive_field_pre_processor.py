# ---   IMPORTS   --- #
# ------------------- #
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
        self.support_chunk_size = kwargs.get('support_chunk_size', 0)
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
            'support_chunk_size': self.support_chunk_size,
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
        self.support_chunk_size = state.get('support_chunk_size', 0)
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
        self.last_call_neighborhoods = None
        self.last_call_neighborhoods = None
