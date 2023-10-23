# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation, EvaluationException
from src.report.meta_report import MetaReport
from src.report.classified_pcloud_report import ClassifiedPcloudReport
from src.report.pwise_activations_report import PwiseActivationsReport


# ---   CLASS   --- #
# ----------------- #
class DLModelEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a deep learning model. See
    :class:`.DLModelEvaluator`.

    :ivar X: The input data.
    :ivar y: Expected values.
    :ivar yhat: Point-wise predictions.
    :ivar zhat: Point-wise outputs (e.g., softmax).
    :ivar activations: Point-wise activations.
    :ivar class_names: The name for each class.
    """
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a DLModelEvaluation.

        :param kwargs: The attributes for the DLModelEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of DLModelEvaluation
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)
        self.yhat = kwargs.get('yhat', None)
        self.zhat = kwargs.get('zhat', None)
        self.activations = kwargs.get('activations', None)
        self.class_names = kwargs.get('class_names', None)
        # Validate attributes
        if self.X is None:
            raise EvaluationException(
                'DLModelEvaluation without a matrix of point coordinates (X) '
                'is not supported.'
            )

    # ---  EVALUATION METHODS  --- #
    # ---------------------------- #
    def report(self, **kwargs):
        """
        Transform the DLModelEvaluation into a DLModelReport.

        See :class:`.DLModelReport`.

        :return: The DLModelReport representing the DLModelEvaluation.
        :rtype: :class:`.DLModelReport`
        """
        # Initialize reports
        reports = []
        # Handle point-wise outputs report
        if self.class_names is not None:
            reports.append({
                'name': 'Point-wise outputs report',
                'report': ClassifiedPcloudReport(
                    X=self.X, y=self.y, yhat=self.yhat, zhat=self.zhat,
                    class_names=self.class_names
                ),
                'path_key': 'path'
            })
        # Handle point-wise activations report
        if self.activations is not None:
            reports.append({
                'name': 'Point-wise activations report',
                'report': PwiseActivationsReport(
                    X=self.X, activations=self.activations, y=self.y
                ),
                'path_key': 'pwise_activations_path'
            })
        # Validate reports
        if len(reports) < 1:
            raise EvaluationException(
                'DLModelEvaluation failed to generate reports.'
            )
        # Return MetaReport
        return MetaReport(reports)

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_report`.
        """
        return self.class_names is not None or self.activations is not None
