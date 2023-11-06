# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException


# ---   CLASS   --- #
# ----------------- #
class TrainingHistoryReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to report (potentially many metrics) the training history of a deep
    learning model, i.e., neural networks.

    :ivar history: The history.
    :vartype history: :class:`tf.keras.callbacks.History`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, history, **kwargs):
        """
        Initialize an instance of TrainingHistoryReport.

        :param history:
        :type history: :class:`tf.keras.callbacks.History`
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the TrainingHistoryReport
        self.history = history
        # Validate attributes
        if self.history is None:
            raise ReportException(
                'TrainingHistoryReport without training history is not '
                'supported. None was given.'
            )
        if self.history.history is None or len(self.history.history) < 1:
            raise ReportException(
                'TrainingHistoryReport received an empty history. It is not '
                'supported.'
            )

    # ---  TO STRING  --- #
    # ------------------- #
    def __str__(self, sep=','):
        """
        The string representation of the Training History report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        hist = self.history.history
        # ---   Header   --- #
        s = f'epoch{sep}{sep.join(hist.keys())}\n'
        # ---   Body   --- #
        vals = list(hist.values())
        num_epochs = len(vals[0])
        vals = [
            [
                f'{vals[j][i]:.6f}' if isinstance(vals[j][i], float)
                else str(vals[j][i])
                for j in range(len(hist))
            ]
            for i in range(num_epochs)
        ]
        for j in range(num_epochs):
            s += f'{(j+1)}{sep}{sep.join(vals[j])}\n'
        # Return
        return s
