# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation
from src.report.classification_uncertainty_report import \
    ClassificationUncertaintyReport


# ---   CLASS   --- #
# ----------------- #
class ClassificationUncertaintyEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating the uncertainty for a given
    classification. See :class:`.ClassificationUncertaintyEvaluator`.

    :ivar class_names: The name for each class.
    :vartype class_names: list
    :ivar X: The matrix with the coordinates of the points.
    :vartype X: :class:`np.ndarray`
    :ivar y: The point-wise classes (reference).
    :vartype y: :class:`np.ndarray`
    :ivar yhat: The point-wise classes (predictions).
    :vartype yhat: :class:`np.ndarray`
    :ivar Zhat: Predicted class probabilities.
    :vartype Zhat: :class:`np.ndarray`
    :ivar pwise_entropy: The point-wise Shannon entropy.
    :vartype pwise_entropy: :class:`np.ndarray`
    :ivar weighted_hspace_entropy: The weighted Shannon entropy based on the
        partition of the feature space into half-spaces.
    :vartype weighted_hspace_entropy: :class:`np.ndarray`
    :ivar class_ambiguity: The point-wise class ambiguity.
    :vartype class_ambiguity: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassificationUncertaintyEvaluation.

        :param kwargs: The attributes for the
            ClassificationUncertaintyEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassificationUncertaintyEvaluation
        self.class_names = kwargs.get('class_names', None)
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)
        self.yhat = kwargs.get('yhat', None)
        self.Zhat = kwargs.get('Zhat', None)
        self.pwise_entropy = kwargs.get('pwise_entropy', None)
        self.weighted_hspace_entropy = kwargs.get(
            'weighted_hspace_entropy', None
        )
        self.class_ambiguity = kwargs.get('class_ambiguity', None)

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the ClassificationUncertaintyEvaluation into a
        ClassificationUncertaintyReport.

        See :class:`.ClassificationUncertaintyReport`

        :return: The ClassificationUncertaintyReport representing the
            ClassificationUncertaintyEvaluation.
        :rtype: :class.`.ClassificationUncertaintyReport`
        """
        return ClassificationUncertaintyReport(
            class_names=self.class_names,
            X=self.X,
            y=self.y,
            yhat=self.yhat,
            Zhat=self.Zhat,
            pwise_entropy=self.pwise_entropy,
            weighted_hspace_entropy=self.weighted_hspace_entropy,
            class_ambiguity=self.class_ambiguity
        )

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_report`.
        """
        return (
            self.X is not None and (
                self.pwise_entropy is not None or
                self.weighted_hspace_entropy is not None or
                self.weighted_hspace_entropy is not None
            )
        )

    def plot(self, **kwargs):
        """
        Transform the ClassificationUncertaintyEvaluation into a
        ClassificationUncertaintyPlot.

        See :class:`.ClassificationUncertaintyPlot`.

        :param kwargs: The key-word arguments for the plot.
        :return: The ClassificationUncertaintyPlot representing the
            ClassificationUncertaintyEvaluation.
        :rtype: :class:`.ClassificationUncertaintyPlot`
        """
        print('ClassificationUncertaintyEvaluation.plot : TODO Implement')
        # TODO Rethink : Implement (histograms by all, by y, and by yhat)

    def can_plot(self):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`
        """
        return (
            self.pwise_entropy is not None or
            self.weighted_hspace_entropy is not None or
            self.weighted_hspace_entropy is not None
        )
