# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.hierarchical_fps_post_processor import \
    HierarchicalFPSPostProcessor


# ---   CLASS   --- #
# ⨪---------------- #
class HierarchicalPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess the output of a hierarchical neural network (e.g., hierarchical
    autoencoders) to transform it to the expected output format.

    :ivar hierarchical_preproc: The preprocessor that generated the input for
        the model which output must be handled by the post-processor.
    :vartype hierarchical_preproc: :class:`.HierarchicalPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, hierarchical_preproc, **kwargs):
        """
        Initialization/instantiation of a hierarchical post-processor.

        :param hierarchical_preproc: The pre-processor associated to the model
            which output must be handled by the post-processor.
        :type hierarchical_preproc: :class:`.HierarchicalPostProcessor`
        :param kwargs: The key-word arguments for the
            HierarchicalPostProcessor.
        """
        # Assign attributes
        self.hierarchical_preproc = hierarchical_preproc  # assoc. pre-proc.
        if self.hierarchical_preproc is None:
            raise DeepLearningException(
                'HierarchicalPostProcessor needs the corresponding '
                'HierarchicalPreProcessor'
            )
        # Handle expected post-processors
        pre_processor_type = self.hierarchical_preproc.pre_processor_type
        if pre_processor_type.lower() == 'hierarchical_fps':
            self.post_processor = HierarchicalFPSPostProcessor(
                self.hierarchical_preproc.pre_processor
            )
        else:  # Unexpected pre-processor
            raise DeepLearningException(
                'HierarchicalPreProcessor received an unexpected '
                'post_processor_type: '
                f'"{self.hierarchical_preproc.pre_processor_type}"'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" give the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R_1` points
            that must be propagated back to the :math:`m` points of the
            original point cloud.
        :type inputs: dict
        :return: The :math:`m` point-wise predictions.
        """
        return self.post_processor(inputs)
