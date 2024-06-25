# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.prediction_reducer import PredictionReducer
from src.utils.preds.sum_pred_reduce_strategy import SumPredReduceStrategy
from src.utils.preds.mean_pred_reduce_strategy import MeanPredReduceStrategy
from src.utils.preds.max_pred_reduce_strategy import MaxPredReduceStrategy
from src.utils.preds.entropic_pred_reduce_strategy import \
    EntropicPredReduceStrategy
from src.utils.preds.argmax_pred_select_strategy \
    import ArgMaxPredSelectStrategy
from src.utils.preds.prediction_reducer import PredictionReducerException


# ---   CLASS   --- #
# ----------------- #
class PredictionReducerFactory:
    """
    :author: Alberto M. Esmoris Pena

    Factory to build instances of :class:`.PredictionReducer`.
    """

    # ---   MAKE METHODS   --- #
    # ------------------------ #
    @staticmethod
    def make_from_dict(spec):
        """
        Make a :class:`.PredictionReducer` from the given dict-like
        specification.

        :param spec: The specification on how to build the prediction reducer.
        :return: The built prediction reducer.
        :rtype: :class:`.PredictionReducer`.
        """
        # Build reduce strategy
        reduce_strategy_spec = spec.get('reduce_strategy', None)
        if reduce_strategy_spec is not None:  # Reduce strategy from spec
            reduce_strategy = PredictionReducerFactory.make_reduce_strategy(
                reduce_strategy_spec
            )
        else:  # Default reduce strategy
            reduce_strategy = MeanPredReduceStrategy()
        # Build select strategy
        select_strategy_spec = spec.get('select_strategy', None)
        if select_strategy_spec is not None:  # Select strategy from spec
            select_strategy = PredictionReducerFactory.make_select_strategy(
                select_strategy_spec
            )
        else:
            select_strategy = ArgMaxPredSelectStrategy()
        # Return prediction reducer
        return PredictionReducer(
            reduce_strategy=reduce_strategy,
            select_strategy=select_strategy
        )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def make_reduce_strategy(spec):
        """
        Make a :class:`.PredReduceStrategy` from the given dict-like
        specification.

        :param spec: The specification on how to build the prediction
            reduce strategy.
        :type spec: dict
        :return: The built prediction reduce strategy
        :rtype: :class:`.PredReduceStrategy`
        """
        strategy = spec['type']
        strategy_low = strategy.lower()
        if strategy_low == 'sumpredreducestrategy':
            return SumPredReduceStrategy()
        elif strategy_low == 'meanpredreducestrategy':
            return MeanPredReduceStrategy()
        elif strategy_low == 'maxpredreducestrategy':
            return MaxPredReduceStrategy()
        elif strategy_low == 'entropicpredreducestrategy':
            return EntropicPredReduceStrategy()
        else:
            raise PredictionReducerException(
                'PredictionReducerFactory.make_reduce_strategy does not '
                f'support the "{strategy}" strategy.'
            )

    @staticmethod
    def make_select_strategy(spec):
        """
        Make a :class:`.PredSelectStrategy` from the given dict-like
        specification.

        :param spec: The specification on how to build the prediction
            select strategy.
        :type spec: dict
        :return: The built prediction select strategy.
        :rtype: :class:`.PredSelectStrategy`
        """
        strategy = spec['type']
        strategy_low = strategy.lower()
        if strategy_low == 'argmaxpredselectstrategy':
            return ArgMaxPredSelectStrategy()
        else:
            raise PredictionReducerException(
                'PredictionReduceFactory.make_select_strategy does not '
                f'support the "{strategy}" strategy.'
            )
