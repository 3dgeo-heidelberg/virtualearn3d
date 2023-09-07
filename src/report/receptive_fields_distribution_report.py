# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldsDistributionReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to the distribution of predicted or
    expected values in the receptive fields.
    See :class:`.Report`

    :ivar y_rf: The expected value for each point for each receptive field.
    :vartype y_rf: :class:`np.ndarray`
    :ivar yhat_rf: The predicted value for each point for each receptive field.
    :vartype yhat_rf: :class:`np.ndarray`
    :ivar class_names: The names representing each class.
    :vartype class_names: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of ReceptiveFieldsDistributionReport.

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *y_rf* (``np.ndarray``) --
                The expected value for each point for each receptive field.
            *   *yhat_rf* (``np.ndarray``) --
                The predicted value for each point for each receptive field.
            *   *class_names* (``np.ndarray``) --
                The name representing each class.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.y_rf = kwargs.get('y_rf', None)
        self.yhat_rf = kwargs.get('yhat_rf', None)
        if self.y_rf is None and self.yhat_rf is None:
            raise ReportException(
                'Receptive field distribution report is not possible without '
                'at least the expected or predicted classes.'
            )
        self.class_names = kwargs.get('class_names', None)
        if self.class_names is None:
            raise ReportException(
                'Receptive field distribution reports needs to receive the '
                'class names at initialization.'
            )

    # ---  TO STRING   --- #
    # -------------------- #
    def __str__(self):
        """
        The string representation of the report on the distribution of the
        many receptive fields.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # ---  Table: header  --- #
        s = 'CLASS                   '
        if self.yhat_rf is not None:
            s += ', ' \
                '      PRED COUNT,    PRED FREQ (%), ' \
                '   PRED RF COUNT, PRED RF FREQ (%)'
        if self.y_rf is not None:
            s += ', ' \
                '      TRUE COUNT,    TRUE FREQ (%), ' \
                '   TRUE RF COUNT, TRUE RF FREQ (%)'
        s += '\n'  # End of header line
        # ---  Table: body  --- #
        num_yhat = np.prod(self.yhat_rf.shape) if self.yhat_rf is not None \
            else 0
        num_yhat_rf = len(self.yhat_rf) if self.yhat_rf is not None else 0
        num_y = np.prod(self.y_rf.shape) if self.y_rf is not None else 0
        num_y_rf = len(self.y_rf) if self.y_rf is not None else 0
        for cidx, cname in enumerate(self.class_names):
            # Start the row with the class name
            s += f'{cname:24.24}'
            # Add columns related to predictions, if available
            if self.yhat_rf is not None:
                # Compute absolute and relative frequencies
                yhat_count, yhat_freq, yhat_rf_count, yhat_rf_freq = \
                    ReceptiveFieldsDistributionReport.count(
                        self.yhat_rf, cidx, num_yhat, num_yhat_rf
                    )
                # Add absolute and relative frequencies to row
                s += f', {yhat_count:16d}, {100*yhat_freq:16.4f}, '\
                    f'{yhat_rf_count:16d}, {100*yhat_rf_freq:16.4f}'
            # Add columns related to expected values, if available
            if self.y_rf is not None:
                # Compute absolute and relative frequencies
                y_count, y_freq, y_rf_count, y_rf_freq = \
                    ReceptiveFieldsDistributionReport.count(
                        self.y_rf, cidx, num_y, num_y_rf
                    )
                # Add absolute and relative frequencies to row
                s += f', {y_count:16d}, {100*y_freq:16.4f}, ' \
                     f'{y_rf_count:16d}, {100*y_rf_freq:16.4f}'
            s += '\n'  # End of line for current row
        # ---  Table: foot  --- #
        s += '\nTOTAL:                  '
        if self.yhat_rf is not None:
            s += f'{num_yhat:16d},              100, '\
                f'{num_yhat_rf:16d},             100'
        if self.y_rf is not None:
            if self.yhat_rf is not None:
                s += ', '
            s += f'{num_y:16d},              100, ' \
                 f'{num_y_rf:16d},             100'
        # Return
        return s

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def count(rf, cidx, num_cases, num_rf):
        """
        Method to count the cases among all receptive fields and also how many
        receptive fields contain at least a single case of current class.

        :param rf: The receptive field (either for expected or predicted
            values).
        :param cidx: The index of the current class to consider for counting.
        :param num_cases: How many total cases, considering all classes, there
            are.
        :param num_rf: How any receptive fields there are.
        :return: The absolute frequency of cases of current class among all
            receptive fields, the corresponding relative frequency, the
            absolute frequency receptive fields containing at least one case
            of current class, and the corresponding relative frequency.
        """
        # Count how many cases of current class
        count = np.count_nonzero(rf == cidx)
        freq = count/num_cases
        # Count how many receptive fields contain the current class
        rf_count = np.count_nonzero([np.any(rf_i == cidx) for rf_i in rf])
        rf_freq = rf_count/num_rf
        # Return
        return count, freq, rf_count, rf_freq
