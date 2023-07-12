# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation, EvaluationException
from src.report.rand_forest_report import RandForestReport
from src.plot.rand_forest_plot import RandForestPlot


# ---   CLASS   --- #
# ----------------- #
class RandForestEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a trained random forest. See
    :class:`.RandForestEvaluator`.

    :ivar problem_name: See :class:`.Evaluator`
    :ivar fnames: The name for each feature.
    :vartype fnames: list or tuple
    :ivar importance: The normalized importance of each feature in [0, 1].
    :vartype importance: :class:`np.ndarray`
    :ivar permutation_importance_mean: The normalized mean permutation
        importance of each feature in [0, 1].
    :vartype permutation_importance_mean: :class:`np.ndarray`
    :ivar permutation_importance_stdev: The standard deviation of the
        normalized permutation importance of each feature.
    :vartype permutation_importance_stdev: :class:`np.ndarray`
    :ivar trees: The list of trees representing the estimators of the random
        forest.
    :vartype trees: list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a RandForestEvaluation.

        :param kwargs: The attributes for the RandForestEvaluation
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of RandForestEvaluation
        self.fnames = kwargs.get('fnames', None)
        self.importance = kwargs.get('importance', None)
        self.permutation_importance_mean = kwargs.get(
            'permutation_importance_mean', None
        )
        self.permutation_importance_stdev = kwargs.get(
            'permutation_importance_stdev', None
        )
        self.trees = kwargs.get('trees', None)
        self.max_tree_depth = kwargs.get('max_tree_depth', 5)
        # Validate
        if (
            self.importance is not None and
            self.permutation_importance_mean is not None and
            len(self.importance) != len(self.permutation_importance_mean)
        ):
            raise EvaluationException(
                'RandForestEvaluation with {m} importances and {n} '
                'permutation importances is not supported.\n'
                'The number of given importances must match the number of '
                'given permutation importances.'.format(
                    m=len(self.importance),
                    n=len(self.permutation_importance_mean)
                )
            )
        if (
            self.permutation_importance_mean is not None and
            self.permutation_importance_stdev is not None and
            len(self.permutation_importance_mean) !=
                len(self.permutation_importance_stdev)
        ):
            raise EvaluationException(
                'RandForestEvaluation with {m} permutation importance means '
                'demands {m} permutation importance standard deviations.\n'
                'However, {n} standard deviations were given.'.format(
                    m=len(self.permutation_importance_mean),
                    n=len(self.permutation_importance_stdev)
                )
            )

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the RandForestEvaluation into a RandForestReport.

        See :class:`.RandForestReport`.

        :return: The RandForestReport representing the RandForestEvaluation.
        :rtype: :class:`.RandForestReport`
        """
        return RandForestReport(
            self.importance,
            permutation_importance_mean=self.permutation_importance_mean,
            permutation_importance_stdev=self.permutation_importance_stdev,
            fnames=self.fnames
        )

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_report`.
        """
        return self.importance is not None and len(self.importance) > 0

    def plot(self, **kwargs):
        r"""
        Transform the RandForestEvaluation into a RandForestPlot.

        See :class:`.RandForestPlot`.

        :Keyword Arguments:
            *   *path* (``str``) --
                The path to store the plot.
            *   *show* (``bool``) --
                Boolean flag to handle whether to show the plot (True) or not
                (False).

        :return: The RandForestPlot representing the RandForestEvaluation.
        :rtype: :class:`.RandForestPlot`
        """
        return RandForestPlot(
            self.trees,
            fnames=self.fnames,
            max_depth=self.max_tree_depth,
            path=kwargs.get('path', None),
            show=kwargs.get('show', False)
        )

    def can_plot(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`.
        """
        return self.trees is not None and len(self.trees) > 0
