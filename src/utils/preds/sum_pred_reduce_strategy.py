# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.pred_reduce_strategy import PredReduceStrategy
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SumPredReduceStrategy(PredReduceStrategy):
    r"""
    :author: Alberto M. Esmoris Pena

    Reduce many predictions per point to a single one by summation.

    The reduced prediction for the :math:`j-th` class of the :math:`i`-th point
    will be as shown below, assuming :math:`K` values for the reduction.

    .. math::
        z_{ij} = \sum_{k=1}^{K}{z_{ijk}}

    See :class:`.PredReduceStrategy`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a mean prediction reduction strategy.

        :param kwargs: The attributes for the MeanPredReduceStrategy.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---  REDUCE METHODS  --- #
    # ------------------------ #
    def reduce(self, reducer, npoints, nvals, Z, I):
        """
        See :class:`.PredReduceStrategy` and
        :meth:`.PredReduceStrategy.reduce`.
        """
        utype = Z[0].dtype
        u = np.zeros((npoints, nvals), dtype=utype) if nvals > 1 \
            else np.zeros(npoints, dtype=utype)
        for i, Zi in enumerate(Z):
            u[I[i]] += Zi
        # Return
        return u
