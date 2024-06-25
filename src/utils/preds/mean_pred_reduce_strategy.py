# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.pred_reduce_strategy import PredReduceStrategy
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor


# ---   CLASS   --- #
# ----------------- #
class MeanPredReduceStrategy(PredReduceStrategy):
    r"""
    :author: Alberto M. Esmoris Pena

    Reduce many predictions per point to a single one by taking the mean value.

    The reduced prediction for the :math:`j-th` class of the :math:`i`-th point
    will be as shown below, assuming :math:`K` values for the reduction.

    .. math::
        z_{ij} = \dfrac{1}{K} \sum_{k=1}^{K}{z_{ijk}}

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
        return GridSubsamplingPostProcessor.pwise_reduce(npoints, nvals, I, Z)
