# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException


# ---   CLASS   --- #
# ----------------- #
class DeepLearningModelSummaryReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports that summarize a deep learning model.
    See :class:`.Report`.
    See also :class:`.SimpleDLModelHandler`.

    :ivar model: The model to be summarized (it does not need to be compiled
        but the architecture must have been built).
    :vartype model: :class:`tf.keras.Model`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, model, **kwargs):
        """
        Initialize an instance of DeepLearningModelSummaryReport.

        :param model: The model to be summarized.
        :type model: :class:`tf.keras.Model`
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the DeepLearningModelSummaryReport
        self.model = model
        # Validate
        if self.model is None:
            raise ReportException(
                'DeepLearningModelSummaryReport needs to know the model to be '
                'summarized.'
            )

    def __str__(self):
        """
        The string representation of the deep learning model summary report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = 'Deep learning model summary:\n\n'
        # ---  Content  --- #
        s_list = []
        self.model.summary(print_fn=lambda x: s_list.append(x))
        s = s + '\n'.join(s_list)
        # Return
        return s
