# ---   EXCEPTIONS   --- #
# ---------------------- #
class EvaluationException(Exception):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to evaluations.
    See :class:`.VL3DException` and :class:`.EvaluationException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Evaluation:
    """
    :author: Alberto M. Esmoris Pena

    Class for evaluation results. See :class:`.Evaluator`.

    :ivar problem_name: The name of the evaluated problem.
    :vartype problem_name: str
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an Evaluation.

        :param kwargs: The attributes for the Evaluation.
        """
        # Fundamental initialization of any evaluation
        self.problem_name = kwargs.get("problem_name", "UNKNOWN PROBLEM")

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the evaluation into a report.

        By default, this method is not supported. Depending on the evaluation
        subclass, this method might be overriden to provide automatic report
        generation.

        :return: The report representing the evaluation.
        :rtype: :class:`.Report`
        """
        raise EvaluationException(
            f'{__class__} does not support automatic report generation.'
        )

    def can_report(self):
        """
        Check whether the evaluation object can generate a report or not.

        :return: True if a report can be generated, False otherwise.
        """
        return False

    def plot(self, **kwargs):
        """
        Transform the evaluation into a plot (or many plots).

        By default, this method is not supported. Depending on the evaluation
        subclass, this method might be overriden to provide automatic plot
        generation.

        :return: The plot representing the evaluation.
        :rtype: :class:`.Plot`
        """
        raise EvaluationException(
            f'{__class__} does not support automatic plot generation.'
        )

    def can_plot(self):
        """
        Check whether the evaluation object can generate a plot or not.

        :return: True if a plot can be generated, False otherwise.
        """
        return False
