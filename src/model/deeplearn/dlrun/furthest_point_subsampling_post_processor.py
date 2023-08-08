# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor
import src.main.main_logger as LOGGING
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess an input in the furthest point subsampling space back to the
    original space before the subsampling.

    See :class:`.FurthestPointSubsamplingPreProcessor`.

    :ivar fps_preproc: The preprocessor that generated the furthest point
        subsampling that must be reverted by the post-processor.
    :vartype fps_preproc: :class:`.FurthestPointSubsamplingPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fps_preproc, **kwargs):
        """
        Initialization/instantiation of a Furthest Point Subsampling
        post-processor.

        :param kwargs: The key-word arguments for the
            FurthestPointSubsamplingPostProcessor.
        """
        # Assign attributes
        self.fps_preproc = fps_preproc
        if self.fps_preproc is None:
            raise DeepLearningException(
                'FurthestPointSubsamplingPostProcessor needs the '
                'corresponding FurthestPointSubsamplingPreProcessor.'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" gives the coordinates
        of the points in the original point cloud. Also, the key "z" gives the
        predictions computed on a receptive field of :math:`R` points that must
        be propagated back to the :math:`m` points of the original point cloud.
        :type inputs: dict
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        # TODO Rethink: Duplicated code wrt GS subsampling
        # Extract inputs
        start = time.perf_counter()
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Transform each prediction by propagation
        rf = self.fps_preproc.last_call_receptive_fields
        I = self.fps_preproc.last_call_neighborhoods
        z_propagated = joblib.Parallel(n_jobs=self.fps_preproc.nthreads)(
            joblib.delayed(
                rfi.propagate_values
            )(
                z_reduced[i], reduce_strategy='mean'
            )
            for i, rfi in enumerate(rf)
        )
        # Reduce point-wise many predictions by computing the mean
        z = GridSubsamplingPostProcessor.pwise_reduce(
            X.shape[0], num_classes, I, z_propagated
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The furthest point subsampling post processor generated {len(z)} '
            f'propagations from {len(z_reduced[0])} reduced predictions '
            f'for each of the {len(z_reduced)} FPS receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return z
