# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class GridSubsamplingPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess an input in the grid subsampling space back to the
    original space before the subsampling.

    See :class:`.GridSubsamplingPreProcessor`.

    :ivar gs_preproc: The preprocessor that generated the grid subsampling
        that must be reverted by the post-processor.
    :vartype gs_preproc: :class:`.GridSubsamplingPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, gs_preproc, **kwargs):
        """
        Initialization/instantiation of a Grid Subsampling post-processor.

        :param kwargs: The key-word arguments for the
            GridSubsamplingPostProcessor.
        """
        # Assign attributes
        self.gs_preproc = gs_preproc
        if self.gs_preproc is None:
            raise DeepLearningException(
                'GridSubsamplingPostProcessor needs the corresponding '
                'GridSubsamplingPreProcessor.'
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
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        start = time.perf_counter()
        z = GridSubsamplingPostProcessor.post_process(
            inputs,
            self.gs_preproc.last_call_receptive_fields,
            self.gs_preproc.last_call_neighborhoods,
            nthreads=self.gs_preproc.nthreads
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The grid subsampling post processor generated {len(z)} '
            f'propagations from {len(inputs["z"][0])} reduced predictions '
            f'for each of the {len(inputs["z"])} GS receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        return z

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def pwise_reduce(npoints, nvars, I, v_propagated):
        """
        Compute a point-wise reduction of propagated values with overlapping.
        In other words, this method can be used to reduce values computed
        on overlapping neighborhoods so there is potentially more than one
        value for the same variable of the same point.

        :param npoints: The number of points.
        :param nvars: The number of considered point-wise variables.
        :param I: The list of neighborhoods. I[i] is the list of indices
            corresponding to the points composing the neighborhood i.
        :param v_propagated: The values to be point-wise reduced. They often
            come from a propagation operation computed on a receptive field,
            thus the name.
        :return: The reduced v vector with a single value for the same variable
            of the same point.
        :rtype: :class:`np.ndarray`
        """
        count = np.zeros(npoints, dtype=int)
        u = np.zeros((npoints, nvars), dtype=float) if nvars > 1 \
            else np.zeros(npoints, dtype=float)
        for i, v_prop_i in enumerate(v_propagated):
            u[I[i]] += v_prop_i
            count[I[i]] += 1
        non_zero_mask = count != 0
        u[non_zero_mask] = \
            u[non_zero_mask] / count[non_zero_mask] if len(u.shape) < 2 \
            else (u[non_zero_mask].T/count[non_zero_mask]).T
        # Return
        return u

    @staticmethod
    def post_process(inputs, rf, I, nthreads=1):
        """
        Computes the post-processing logic. The method is used to aid the
        :meth:`grid_subsampling_post_processor.GridSubsamplingPostProcessor.__call__`
        method.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R` points
            that must be propagated back to the :math:`m` points of the
            original point cloud.
        :type inputs: dict
        :param rf: The receptive fields to compute the propagations. See
            :class:`.ReceptiveField` and :class:`.ReceptiveFieldGS`.
        :type rf: list
        :param I: The list of neighborhods, where each neighborhood is given
            as a list of indices.
        :type I: list
        :param nthreads: The number of threads for parallel computing.
        :type nthreads: int
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        # Extract inputs
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Transform each prediction by propagation
        z_propagated = joblib.Parallel(n_jobs=nthreads)(
            joblib.delayed(
                rfi.propagate_values
            )(
                z_reduced[i], reduce_strategy='mean'
            )
            for i, rfi in enumerate(rf)
        )
        # Reduce point-wise many predictions by computing the mean
        return GridSubsamplingPostProcessor.pwise_reduce(
            X.shape[0], num_classes, I, z_propagated
        )
