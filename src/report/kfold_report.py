# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report


# ---   CLASS   --- #
# ----------------- #
class KFoldReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to k-folding.
    See :class:`.Report`.
    See also :class:`.Model`, :meth:`model.Model.train_stratified_kfold,
    :class:`.KFoldEvaluator`, and :class:`.KFoldEvaluation`.

    :ivar problem_name: See :class:`.Evaluator`
    :ivar metric_names: See :class:`.KFoldEvaluation`
    :ivar mu: See :class:`.KFoldEvaluation`
    :ivar sigma: See :class:`.KFoldEvaluation`
    :ivar Q: See :class:`.KFoldEvaluation`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, mu, sigma, Q, **kwargs):
        """
        Initialize an instance of KFoldReport.

        :param mu: See :class:`.KFoldEvaluation`
        :param sigma: See :class:`.KFoldEvaluation`
        :param Q: See :class:`.KFoldEvaluation`
        :param kwargs: The key-word arguments defining the report's
            attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the KFoldReport
        self.problem_name = kwargs.get('problem_name', 'UNKNOWN PROBLEM')
        self.metric_names = kwargs.get('metric_names', None)
        self.mu = mu
        self.sigma = sigma
        self.Q = Q
        # Handle serial metric_names when None are given
        if self.metric_names is None:
            self.metric_names = [f'METRIC_{i}' for i in range(len(self.mu))]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the k-fold procedure report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = 'k-fold report on {pn} over {m} metrics:\n'.format(
            pn=self.problem_name,
            m=len(self.mu)
        )
        # ---  Head  --- #
        s += '          '
        for mname in self.metric_names:
            s += f' {mname:12.12}'
        # ---  Body  --- #
        s += '\nmean      '
        for mu in self.mu:
            s += f' {100*mu:10.3f}  '
        s += '\nstdev     '
        for sigma in self.sigma:
            s += f' {100*sigma:10.3f}  '
        s += '\nQ1        '
        for q1 in self.Q[0]:
            s += f' {100*q1:10.3f}  '
        s += f'\nQ{self.Q.shape[0]}        '
        for qn in self.Q[-1]:
            s += f' {100*qn:10.3f}  '
        # Return
        return s
