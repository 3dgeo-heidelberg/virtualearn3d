# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation
from src.report.classification_report import ClassificationReport
from src.plot.classification_plot import ClassificationPlot


# ---   CLASS   --- #
# ----------------- #
class ClassificationEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a classification. See
    :class:`.ClassificationEvaluator`.

    :ivar class_names: See :class:`ClassificationEvaluator`.
    :ivar metric_names: See :class:`ClassificationEvaluator`.
    :ivar class_metric_names: See :class:`ClassificationEvaluator`.
    :ivar yhat_count: The count of cases per predicted label.
    :vartype yhat_count: :class:`np.ndarray`
    :ivar y_count: The count of cases per expected label (real class
        distribution).
    :vartype y_count: :class:`np.ndarray`
    :ivar conf_mat: The confusion matrix where rows are the expected or true
        labels and columns are the predicted labels.
    :vartype conf_mat: :class:`np.ndarray`
    :ivar metric_scores: The score for each metric, i.e., metric_scores[i] is
        the computed score corresponding to metric_names[i].
    :vartype metric_scores: :class:`np.ndarray`
    :ivar class_metric_scores: The class-wise scores for each metric.
        class_metric_scores[i][j] is the metric i calculated for the class j.
    :vartype class_metric_scores: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassificationEvaluation.

        :param kwargs: The attributes for the ClassificationEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassificationEvluation
        self.class_names = kwargs.get('class_names', None)
        self.metric_names = kwargs.get('metric_names', None)
        self.class_metric_names = kwargs.get('class_metric_names', None)
        self.yhat_count = kwargs.get('yhat_count', None)
        self.y_count = kwargs.get('y_count', None)
        self.conf_mat = kwargs.get('conf_mat', None)
        self.metric_scores = kwargs.get('metric_scores', None)
        self.class_metric_scores = kwargs.get('class_metric_scores', None)

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the ClassificationEvaluation into a ClassificationReport.

        See :class:`.ClassificationReport`.

        :return: The ClassificationReport representing the
            ClassificationEvaluation.
        :rtype: :class:`.ClassificationReport`
        """
        return ClassificationReport(
            class_names=self.class_names,
            yhat_count=self.yhat_count,
            y_count=self.y_count,
            conf_mat=self.conf_mat,
            metric_names=self.metric_names,
            metric_scores=self.metric_scores,
            class_metric_names=self.class_metric_names,
            class_metric_scores=self.class_metric_scores
        )

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_report`.
        :return:
        """
        return (
            (self.yhat_count is not None and self.y_count is not None) or
            self.conf_mat is not None or
            self.metric_scores is not None or
            self.class_metric_scores is not None
        )

    def plot(self, **kwargs):
        r"""
        Transform the ClassificationEvaluation into a ClassificationPlot.

        See :class:`.ClassificationPlot`.

        :param kwargs:
        :return: The ClassificationPlot representing the
            ClassificationEvaluation.
        :rtype: :class:`.ClassificationEvaluation`
        """
        return ClassificationPlot(
            class_names=self.class_names,
            yhat_count=self.yhat_count,
            y_count=self.y_count,
            conf_mat=self.conf_mat,
            path=kwargs.get(
                'path',
                kwargs.get('confusion_matrix_path', None)
            ),
            class_distribution_path=kwargs.get(
                'class_distribution_path', None
            ),
            show=kwargs.get('show', False)
        )

    def can_plot(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`.
        """
        return (
            (self.yhat_count is not None and self.y_count is not None) or
            self.conf_mat is not None
        )
