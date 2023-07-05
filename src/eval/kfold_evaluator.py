# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.kfold_evaluation import KFoldEvaluation
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class KFoldEvaluator(Evaluator):
    """
    :author: Alberto M. Esmoris Pena

    Class to evaluate kfold procedures. See :class:`.KFoldEvaluation`.

    :ivar quantile_cuts: The cut points defining the quantiles. By default,
        they represent the quartiles, i.e., 1/4, 2/4, 3/4.
    :vartype quantile_cuts: list or tuple or np.ndarray
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a KFoldEvaluator.

        :param kwargs: The attributes for the KFoldEvaluator.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of KFoldEvaluator
        self.quantile_cuts = kwargs.get(
            'quantile_cuts',
            [i/4 for i in range(1, 4)]
        )
        self.metric_names = kwargs.get(
            'metric_names', None
        )

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, X, **kwargs):
        """
        Evaluate the results of a k-folding procedure.

        :param X: The matrix of quantitative evaluations. Each row must
            represent a fold and each column an evaluation metric.
            Thus, X[i][j] is the j-th evaluation metric on the i-th fold.
        :return: The evaluation of the k-folding.
        :rtype: :class:`.KFoldEvaluation`
        """
        # Extract matrix of quantitative evaluations
        if X is None:
            raise EvaluatorException(
                'KFoldEvaluator cannot evaluate without an input matrix of '
                'quantitative evaluations. None was given.'
            )
        # Evaluate
        return KFoldEvaluation(
            problem_name=self.problem_name,
            metric_names=self.metric_names,
            X=X,
            mu=np.mean(X, axis=0),
            sigma=np.std(X, axis=0),
            Q=np.quantile(X, self.quantile_cuts, axis=0)
        )
