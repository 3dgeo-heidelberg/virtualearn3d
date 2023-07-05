# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


# ---   EXCEPTIONS   --- #
# ---------------------- #
class EvaluatorException(Exception):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to evaluators.
    See :class:`.VL3DException` and :class:`.EvaluationException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Evaluator:
    """
    :author: Alberto M. Esmoris Pena

    Class for evaluation operations. See :class:`.Evaluation`.

    :ivar problem_name: The name of the problem that is being evaluated.
    :vartype problem_name: str
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an Evaluator.

        :param kwargs: The attributes for the Evaluator.
        """
        # Fundamental initialization of any evaluator
        self.problem_name = kwargs.get("problem_name", "UNKNOWN PROBLEM")

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    @abstractmethod
    def eval(self, x, **kwargs):
        """
        Evaluate something and yield an evaluation.

        :param x: The input to be evaluated.
        :return: Evaluation.
        :rtype: :class:`.Evaluation`
        """
        pass
