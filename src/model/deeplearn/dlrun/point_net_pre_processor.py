# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor import\
    FurthestPointSubsamplingPreProcessor


# ---   CLASS   --- #
# ----------------- #
class PointNetPreProcessor:
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into the PointNet neural network.

    :ivar pre_processor_type: String representing the type of pre-processor
        to generate the Input for the PointNet neural network.
    :vartype pre_processor_type: str
    :ivar pre_processor: The pre-processor itself (instantiated).
    :vartype pre_processor: :class:`.GridSubsamplingPreProcessor` or
        :class:`.FurthestPointSubsamplingPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a PointNet pre-processor.

        :param kwargs: The key-word arguments for the PointNetPreProcessor.
        """
        # Assign attributes
        self.pre_processor_type = kwargs.get('pre_processor', None)
        if self.pre_processor_type is None:
            raise DeepLearningException(
                'PointNetPreProcessor needs a pre_processor specification.'
            )
        # Handle expected pre-processors
        if self.pre_processor_type.lower() == 'grid_subsampling':
            self.pre_processor = GridSubsamplingPreProcessor(**kwargs)
        elif self.pre_processor_type.lower() == 'furthest_point_subsampling':
            self.pre_processor = FurthestPointSubsamplingPreProcessor(**kwargs)
        else:  # Unexpected pre-processor
            raise DeepLearningException(
                'PointNetPreprocessor received an unexpected pre_processor_'
                f'_type: "{self.pre_processor_type}"'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the pre-processing logic.

        See :class:`.GridSubsamplingPreProcessor` and
        :class:`.FurthestPointSubsamplingPreProcessor`.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a PointNet model.
        :type inputs: dict
        :return: Either (Xout, yout) or Xout. Where Xout are the points
            representing the receptive field and yout (only given when
            "y" was given in the inputs dictionary) the corresponding
            reference values for those points.
        """
        # TODO Rethink : The solution of taking the coordinates only is not
        # admissible because features must be reduced and propagated
        # through the receptive fields too
        X = inputs['X']
        if isinstance(X, list):  # If many inputs, take the coordinates only
            X = X[0]  # Coordinates are assumed to be the first array
        _inputs = {'X': X, 'y': inputs['y']}
        return self.pre_processor(_inputs)

    # ---   POINT-NET METHODS   --- #
    # ----------------------------- #
    def get_num_input_points(self):
        """
        PointNet pre-processors must provide the expected number of input
        points to batch the many input neighborhoods.

        :return: Number of input points per neighborhood.
        :rtype: int
        """
        return self.pre_processor.get_num_input_points()

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
