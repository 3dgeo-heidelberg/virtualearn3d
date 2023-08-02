# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPostProcessor:
    """
    # TODO Rethink : Doc
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fps_preproc, **kwargs):
        """
        # TODO Rethink : Doc
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
        # TODO Rethink : Doc
        :param inputs:
        :return:
        """
        # TODO Rethink: Duplicated code wrt GS subsampling
        # Extract inputs
        start = time.perf_counter()
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Transform each prediction by propagation
        z_propagated = []
        rf = self.fps_preproc.last_call_receptive_fields
        I = self.fps_preproc.last_call_neighborhoods
        for i, rfi in enumerate(rf):
            z_propagated.append(rfi.propagate_values(
                z_reduced[i], reduce_strategy='mean'
            ))
        # Reduce point-wise many predictions by computing the mean
        count = np.zeros(X.shape[0], dtype=int)
        z = np.zeros((X.shape[0], num_classes), dtype=float)
        for i, z_prop_i in enumerate(z_propagated):
            z[I[i]] += z_prop_i
            count[I[i]] += 1
        z = z / count if len(z.shape) < 2 else (z.T/count).T
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The furthest point subsampling post processor generated {len(z)} '
            f'propagations from {len(z_reduced[0])} reduced predictions '
            f'for each of the {len(z_reduced)} FPS receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return z
