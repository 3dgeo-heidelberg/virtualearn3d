# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import src.main.main_logger as LOGGING


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

    def __call__(self, x, **kwargs):
        """
        Evaluate with extra logic that is convenient for pipeline-based
        execution.

        See :meth:`evaluator.Evaluator.eval`.
        """

        # Obtain evaluation
        ev = self.eval(x, **kwargs)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_report():
            report = ev.report()
            LOGGING.LOGGER.info(report.to_string())
            report_path = kwargs.get('report_path', None)
            if report_path is not None:
                report.to_file(report_path, out_prefix=out_prefix)
        if ev.can_plot():
            plot_path = kwargs.get('plot_path', None)
            if plot_path is not None:
                ev.plot(path=plot_path).plot(out_prefix=out_prefix)
