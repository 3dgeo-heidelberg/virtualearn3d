# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation, EvaluationException
from src.report.kfold_report import KFoldReport
from src.plot.kfold_plot import KFoldPlot


# ---   CLASS   --- #
# ----------------- #
class KFoldEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a kfold procedure. See
    :class:`.KFoldEvaluator`.

    :ivar problem_name: See :class:`.Evaluator`
    :ivar metric_names: The name for each metric, i.e., the name of each
        component in mu or sigma vectors.
    :vartype metric_names: list or tuple
    :ivar mu: The vector of means such that each component represents the mean
        value of an evaluation metric used to assess the k-folding procedure.
    :vartype mu: :class:`np.ndarray` vector-like
    :ivar sigma: The vector of standard deviations such that each component
        represents the standard deviation of an evaluation metric used to
        assess the k-folding procedure.
    :vartype sigma: :class:`np.ndarray` vector-like
    :ivar Q: The matrix of quantiles such that the component of each column
        vector represents the quantiles of an evaluation metrics used
        to assess the k-folding procedure.
    :vartype Q: :class:`np.ndarray` matrix-like
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a KFoldEvaluation.

        :param kwargs: The attributes for the KFoldEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of KFoldEvaluation
        self.metric_names = kwargs.get('metric_names', None)
        self.X = kwargs.get('X', None)
        self.mu = kwargs.get('mu', None)
        # Validate attributes
        if self.mu is None:
            raise EvaluationException(
                'KFoldEvaluation without mu (mean) is not supported.'
            )
        self.sigma = kwargs.get('sigma', None)
        if self.sigma is None:
            raise EvaluationException(
                'KFoldEvaluation without sigma (standard deviation) is not '
                'supported.'
            )
        self.Q = kwargs.get('Q', None)
        if self.Q is None:
            raise EvaluationException(
                'KFoldEvaluation without Q (quantiles) is not supported.'
            )
        # Validate dimensionality
        if self.X is not None and self.mu is not None:
            if len(self.mu) != self.X.shape[1]:
                raise EvaluationException(
                    'KFoldEvaluation for {m} means and {n} evaluation metrics '
                    'is not supported. The dimensionality of the vector of '
                    'means must match the number of columns of the evaluation '
                    'matrix.'.format(
                        m=len(self.mu),
                        n=self.X.shape[1]
                    )
                )
        if len(self.mu) != len(self.sigma):
            raise EvaluationException(
                'KFoldEvaluation for {m} means and {n} standard deviations '
                'is not supported. The dimensionality of both vectors must be '
                'the same.'.format(
                    m=len(self.mu),
                    n=len(self.sigma)
                )
            )

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the KFoldEvaluation into a KFoldReport.

        See :class:`.KFoldReport`.

        :return: The KFoldReport representing the KFoldEvaluation.
        :rtype: :class:`.KFoldReport`
        """
        return KFoldReport(
            self.mu, self.sigma, self.Q,
            problem_name=self.problem_name,
            metric_names=self.metric_names
        )

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_report`.
        """
        return True

    def plot(self, **kwargs):
        r"""
        Transform the KFoldEvaluation into a KFoldPlot.

        See :class:`.KFoldPlot`.

        :Keyword Arguments:
            *   *path* (``str``) --
                The path to store the plot.
            *   *show* (``bool``) --
                Boolean flag to handle whether to show the plot (True) or not
                (False).

        :return: The KFoldPlot representing the KFoldEvaluation.
        :rtype: :class:`.KFoldPlot`
        """
        return KFoldPlot(
            self.X,
            self.sigma,
            metric_names=self.metric_names,
            path=kwargs.get('path', None),
            show=kwargs.get('show', False)
        )

    def can_plot(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`.
        """
        return len(self.mu) > 1
