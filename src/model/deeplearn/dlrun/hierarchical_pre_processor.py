# TODO Rethink : Remove this class, no longer used?
# TODO Rethink : Better, it should be used as a wrapper for
# HierarchicalFPSPreProcessor as PointNetPreProcessor is to others.
# Also, do a wrapper for post-processing.


# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor


# ---   CLASS   --- #
# ----------------- #
class HierarchicalPreProcessor:
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into a hierarchical neural network
    (e.g., hierarchical autoencoder).

    :ivar pre_processor_type: String representing the type of pre-processor
        to generate the Input for the hierarchical neural network.
    :vartype pre_processor_type: str
    :ivar pre_processor: The pre-processor itself (instantiated).
    :vartype pre_processor: :class:`.HierarchicalFPSPreProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a hierarchical pre-processor.

        :param kwargs: The key-word arguments for the HierarchicalPreProcessor.
        """
        # Assign attributes
        self.pre_processor_type = kwargs.get('pre_processor', None)
        if self.pre_processor_type is None:
            raise DeepLearningException(
                'HierarchicalPreProcessor needs a pre_processor specification.'
            )
        # Handle expected pre-processors
        if self.pre_processor_type.lower() == 'hierarchical_fps':
            self.pre_processor = HierarchicalFPSPreProcessor(**kwargs)
        else:  # Unexpected pre-processor
            raise DeepLearningException(
                'HierarchicalPreProcessor received an unexpected '
                f'pre_processor_type: "{self.pre_processor_type}"'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the pre-processing logic.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a hierarchical model.
        :type inputs: dict
        :return: Either (X, F, ...NDi..., ...Ni..., ...NUi... yout) or
            (X, F, ...NDi..., ...Ni..., ...NUi...). Where X are the input
            points, F are the input features, NDi are point-wise downsamlping
            neighborhoods at depth i, Ni are the point-wise neighborhoods at
            depth i, and NUi are the point-wise upsampling neighborhoods at
            depth i.
        """
        return self.pre_processor(inputs)

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        Assist the :meth:`model.Model.overwrite_pretrained_model` method
        through assisting the
        :meth:`architecture.Architecture.overwrite_pretrained_model` method.

        :param spec: The key-word specification containing the model's
            arguments.
        :type spec: dict
        """
        # Overwrite the attributes of the pre-processor
        if hasattr(self.pre_processor, 'overwrite_pretrained_model'):
            self.pre_processor.overwrite_pretrained_model(spec)
