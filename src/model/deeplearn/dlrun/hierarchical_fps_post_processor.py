# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class HierarchicalFPSPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess an input in the first level of the FPS hierarchy back to the
    original space.

    See :class:`.HierarchicalFPSPreProcessor` and
    :class:`.FurthestPointSubsamplingPostProcessor`.

    :ivar hfps_preproc: The preprocessor that generated the furthest point
        subsampling that must be reverted by the post-processor.
    :vartype hfps_preproc: :class:`.HierarchicalFPSPreProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, hfps_preproc, **kwargs):
        """
        Initialization/instantiation of a hierarchical FPS post-processor.

        :param hfps_preproc: The corresponding hierarchical FPS pre-processor.
        :param kwargs: The key-word arguments for the
            HierarchicalFPSPostProcessor.
        """
        # Assign attributes
        self.hfps_preproc = hfps_preproc
        if self.hfps_preproc is None:
            raise DeepLearningException(
                'HierarchicalFPSPostProcessor needs the '
                'corresponding HierarchicalFPSPreProcessor.'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R_1` points
            (i.e., at depth :math:`d=1`) that must be propagated back to the
            :math:`m` points of the original point cloud.
        :type inputs: dict
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        start = time.perf_counter()
        _inputs = inputs
        if isinstance(inputs['X'], list):
            _inputs = {
                'X': inputs['X'][0],
                'z': inputs['z']
            }
        z = GridSubsamplingPostProcessor.post_process(
            _inputs,
            self.hfps_preproc.last_call_receptive_fields,
            self.hfps_preproc.last_call_neighborhoods,
            nthreads=self.hfps_preproc.nthreads
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The hierarchical FPS post processor generated {len(z)} '
            f'propagations from {len(inputs["z"][0])} reduced predictions '
            f'for each of the {len(inputs["z"])} FPS receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        return z
