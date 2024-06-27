# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class PredReduceStrategy:
    """
    :author: Alberto M. Esmoris Pena

    Interface for reduce operations on predictions.

    See :class:`.PredictionReducer`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a prediction reduction strategy.

        :param kwargs: The attributes for the PredReduceStrategy.
        """
        pass

    # ---  REDUCE METHODS  --- #
    # ------------------------ #
    @abstractmethod
    def reduce(self, reducer, npoints, nvals, Z, I):
        r"""
        The method that provides the logic to reduce potentially more
        predictions than points such that there is at most one prediction
        per point. It must be overridden by any concrete implementation of a
        prediction reduction strategy.

        :param reducer: The prediction reducer that is doing the reduction.
        :type reducer: :class:`.PredictionReducer`
        :ivar npoints: The number of points to which the predictions must be
            reduced to.
        :vartype npoints: int
        :ivar nvals: How many point-wise values must be considered for each point
            (classes for classification tasks, reference values for regression
            tasks).
        :vartype nvals: int
        :param Z: List of arrays. There is one array per input neighborhood,
            each array is a matrix where the rows represent the points and the
            columns the class-wise likelihoods.
        :type Z: list of :class:`np.ndarray`
        :param I: The indices representing the neighborhoods. There are as many
            lists as input neighborhoods. Each list contains the indices
            representing the points in the original point cloud. For example,
            the j-th element of the i-th list I[i][j] is the index of the
            j-th point in the i-th input neighborhood in the original
            point cloud.
        :type I: list of list of int
        :return: The reduced predictions as a matrix
            :math:`\pmb{Z} \in \mathbb{R}^{m \times n_v}` where the :math:`m`
            rows represent the points (I[i][j] will point to a row) and the
            :math:`n_v` columns represent the likelihood for each of the
            predicted classes (classification) or the reference point-wise
            values (regression).
        :rtype: :class:`np.ndarray`
        """
        pass
