# ---   IMPORTS   --- #
# ------------------- #
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PointNetPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess the output of a PointNet neural network to transform it to the
    expected output format.

    :ivar pnet_preproc: The preprocessor that generated the input for the model
        which output must be handled by the post-processor.
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
        # Extract inputs
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Transform each prediction by propagation
        z_propagated = []
        rf = self.pnet_preproc.last_call_receptive_fields
        I = self.pnet_preproc.last_call_neighborhoods
        for i, rfi in enumerate(rf):
            z_propagated.append(rfi.propagate_values(z_reduced[i]))
        # Reduce point-wise many predictions by computing the mean
        count = np.zeros(X.shape[0], dtype=int)
        z = np.zeros((X.shape[0], num_classes), dtype=float)
        for i, z_prop_i in enumerate(z_propagated):
            z[I[i]] += z_prop_i
            count[I[i]] += 1
        z = z / count if len(z.shape) < 2 else (z.T/count).T
        # Return
        return z
