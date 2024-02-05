# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_post_processor \
    import FurthestPointSubsamplingPostProcessor


# ---   CLASS   --- #
# ----------------- #
class PointNetPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess the output of a PointNet neural network to transform it to the
    expected output format.

    :ivar pnet_preproc: The preprocessor that generated the input for the model
        which output must be handled by the post-processor.
    :vartype pnet_preproc: :class:`.PointNetPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, pnet_preproc, **kwargs):
        """
        Initialization/instantiation of a PointNet post-processor.

        :param pnet_preproc: The pre-processor associated to the model which
            output must be handled by the post-processor.
        :type pnet_preproc: :class:`.PointNetPreProcessor`
        :param kwargs: The key-word arguments for the PointNetPostProcessor.
        """
        # Assign attributes
        self.pnet_preproc = pnet_preproc  # Corresponding pre-processor
        if self.pnet_preproc is None:
            raise DeepLearningException(
                'PointNetPostProcessor needs the corresponding '
                'PointNetPreProcessor.'
            )
        # Handle expected post-processors
        if self.pnet_preproc.pre_processor_type == 'grid_subsampling':
            self.post_processor = GridSubsamplingPostProcessor(
                self.pnet_preproc.pre_processor
            )
        elif self.pnet_preproc.pre_processor_type == 'furthest_point_subsampling':
            self.post_processor = FurthestPointSubsamplingPostProcessor(
                self.pnet_preproc.pre_processor
            )
        else:
            raise DeepLearningException(
                'PointNetPostProcessor received an unexpected pre_processor_'
                f'_type: "{self.pnet_preproc.pre_processor_type}"'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R` points
            that must be propagated back to the :math:`m` points of the
            original point cloud.
        :type inputs: dict
        :return: The :math:`m` point-wise predictions.
        """
        return self.post_processor(inputs)
