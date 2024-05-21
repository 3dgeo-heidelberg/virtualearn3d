# ---   IMPORTS   --- #
# ------------------- #
from src.model.tdcomp.training_data_component import TrainingDataComponent
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ClasswiseSampler(TrainingDataComponent):
    """
    :author: Alberto M. Esmoris Pena

    Training data component based on sampling from the input training data to
    satisfy a given class-wise distribution.

    The model arguments of a Classwise sampler are:

    -- ``target_class_distribution`` - `list of int`
        The target distribution for each class.

    -- ``replace`` - `bool`
        Whether to compute the sampling with replacement (i.e., repeating
        data points is allowed) or not.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the class-wise sampler training data component.

        :param kwargs: The attributes for the class-wise training data
            component.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign class-wise sampler attributes
        self.classes = [
            cidx for cidx in range(
                len(self.component_args['target_class_distribution'])
            )
        ]

    # ---   CALL   --- #
    # ---------------- #
    def __call__(self, X, y):
        """
        Apply the class-wise sampler to transform the input training data.

        See :meth:`.TrainingDataComponent.__call__`.
        """
        m = X.shape[0]
        start = time.perf_counter()
        X, y = self.compute_classwise_sampling(X, y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClasswiseSampler applied to {m} training data points yielded '
            f'{X.shape[0]} training data points in '
            f'{end-start:.3f} seconds.'
        )
        return X, y

    def compute_classwise_sampling(self, X, y):
        """
        Compute the class-wise sampling on the given input.

        :param X: The features of the training data points.
        :param y: The expected classes of the training data points.
        :return: The transformed training data.
        :rtype: tuple (X, y)
        """
        Iselected = []
        target_cdistr = self.component_args['target_class_distribution']
        replace = self.component_args['replace']
        for cidx in self.classes:
            Ic = np.flatnonzero(y == cidx)  # Indices of points for cidx class
            target_count = target_cdistr[cidx] if replace else min(
                target_cdistr[cidx], len(Ic)
            )
            Iselected.extend(np.random.choice(
                Ic,
                target_count,
                replace=replace
            ))
        return X[Iselected], y[Iselected]
