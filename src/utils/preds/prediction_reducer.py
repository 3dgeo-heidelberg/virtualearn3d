# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.utils.preds.mean_pred_reduce_strategy import MeanPredReduceStrategy
from src.utils.preds.argmax_pred_select_strategy \
    import ArgMaxPredSelectStrategy


# ---   EXCEPTIONS   --- #
# ---------------------- #
class PredictionReducerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to the reduction of many predictions for the
    same point.
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PredictionReducer:
    r"""
    :author: Alberto M. Esmoris Pena

    Class for reduce and select-after-reduce operations on predictions.

    See :class:`.PredReduceStrategy` and :class:`.PredSelectStrategy`.

    :ivar reduce_strategy: The strategy for the reduction itself.
    :vartype reduce_strategy: :class:`.PredReduceStrategy`
    :ivar select_strategy: The strategy for the selection after the reduction.
    :vartype select_strategy: :class:`.PredSelectStrategy`
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a PredictionReducer.

        :param kwargs: The attributes for the PredictionReducer.
        """
        # Fundamental initialization of any prediction reducer
        self.reduce_strategy = kwargs.get(
            'reduce_strategy',
            MeanPredReduceStrategy()
        )
        self.select_strategy = kwargs.get(
            'select_strategy',
            ArgMaxPredSelectStrategy()
        )
        # Validate strategies
        if self.reduce_strategy is None:
            raise PredictionReducerException(
                'PredictionReducer MUST have a reduce strategy.'
            )
        if not hasattr(self.reduce_strategy, 'reduce'):
            raise PredictionReducerException(
                'PredictionReducer needs a reduce strategy with a reduce '
                f'method but {self.reduce_strategy.__class__.__name__} does '
                'not have one.'
            )
        if self.select_strategy is None:
            raise PredictionReducerException(
                'PredictionReducer MUST have a select strategy.'
            )
        if not hasattr(self.select_strategy, 'select'):
            raise PredictionReducerException(
                'PredictionReducer needs a select strategy with a select '
                f'method but {self.select_strategy.__class__.__name__} does '
                'not have one.'
            )

    # ---  REDUCE METHODS  --- #
    # ------------------------ #
    def reduce(self, npoints, nvals, Z, I):
        """
        See :class:`.PredReduceStrategy` and
        :meth:`.PredReduceStrategy.reduce`.
        """
        return self.reduce_strategy.reduce(self, npoints, nvals, Z, I)

    # ---  SELECT METHODS  --- #
    # ------------------------ #
    def select(self, Z):
        """
        See :class:`.PredSelectStrategy` and :meth:`PredSelectStrategy.select`.
        """
        return self.select_strategy.select(self, Z)
