# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
import src.main.main_logger as LOGGING
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class ClassificationReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to classifications.
    See :class:`.Report`.
    See also :class:`.ClassificationModel`.

    :ivar class_names: See :class:`.ClassificationEvaluation`.
    :ivar yhat_count: See :class:`.ClassificationEvaluation`.
    :ivar y_count: See :class:`.ClassificationEvaluation`.
    :ivar conf_mat: See :class:`.ClassificationEvaluation`.
    :ivar metric_names: See :class:`.ClassificationEvaluation`.
    :ivar metric_scores: See :class:`.ClassificationEvaluation`.
    :ivar class_metric_names: See :class:`.ClassificationEvaluation`.
    :ivar class_metric_scores: See :class:`.ClassificationEvaluation`.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassificationReport.

        :param kwargs: The key-word arguments defining the report's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the ClassificationReport
        self.class_names = kwargs.get('class_names', None)
        self.yhat_count = kwargs.get('yhat_count', None)
        self.y_count = kwargs.get('y_count', None)
        self.conf_mat = kwargs.get('conf_mat', None)
        self.metric_names = kwargs.get('metric_names', None)
        self.metric_scores = kwargs.get('metric_scores', None)
        self.class_metric_names = kwargs.get('class_metric_names', None)
        self.class_metric_scores = kwargs.get('class_metric_scores', None)
        # Handle serial class_names when None are given
        if self.class_names is None and self.yhat_count is not None:
            self.class_names = [
                f'Class{i}' for i in range(len(self.yhat_count))
            ]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the report about a classification.
        See :class:`.Report` and also :meth:`report.Report.__str__`
        """
        # Initialize
        s = '\n    Classification report\n' \
            '=============================\n'
        # Fill with available information
        if self.has_global_eval_info():
            s += '\n'+self.to_global_eval_string()+'\n'
        if self.has_class_eval_info():
            s += '\n'+self.to_class_eval_string()+'\n'
        if self.has_confusion_matrix():
            s += '\n'+self.to_confusion_matrix_string()+'\n'
        if self.has_class_distribution_info():
            s += '\n'+self.to_class_distribution_string()+'\n'
        # Return
        return s

    def to_global_eval_string(self):
        """
        Generate the string representing the classification report with respect
        to the global evaluation.

        :return: String representing the classification report with respect to
        the global evaluation.
        """
        # --- Introduction --- #
        s = 'Global classification evaluation:\n'
        # ---  Head  --- #
        for mname in self.metric_names:
            s += f'{mname:>11.11},'
        s = s[:-1] + '\n'
        # ---  Body  --- #
        for score in self.metric_scores:
            s += f' {100*score:10.3f},'
        s = s[:-1]
        # Return
        return s

    def to_class_eval_string(self):
        """
        Generate the string representing the classification report with respect
        to the class-wise evaluation.

        :return: String representing the classification report with respect to
        the class-wise evaluation.
        """
        # --- Introduction --- #
        s = 'Class-wise classification evaluation:\n'
        # ---  Head  --- #
        s += '                 '
        for mname in self.class_metric_names:
            s += f'{mname:>11.11},'
        s  = s[:-1]
        # ---  Body  --- #
        for i, class_name in enumerate(self.class_names):
            s += f'\n{class_name:16.16} '
            for class_score in self.class_metric_scores[:, i]:
                s += f' {100*class_score:10.3f},'
            s  = s[:-1]
        # Return
        return s

    def to_confusion_matrix_string(self):
        """
        Generate the string representing the classification report with respect
        to the confusion matrix.

        :return: String representing the classification report with respect to
        the confusion matrix.
        """
        # --- Introduction --- #
        s = 'Confusion matrix (rows are true labels, columns are predictions):'
        # ---  Matrix  --- #
        nrows, ncols = self.conf_mat.shape
        s += '\n'
        for i in range(nrows):
            for j in range(ncols):
                s += f'{self.conf_mat[i, j]:9}, '
            s = s[:-2] + '\n'
        # Return
        return s

    def to_class_distribution_string(self):
        """
        Generate the string representing the classification report with respect
        to the class distribution.

        :return: String representing the classification report with respect to
        the class distribution.
        """
        # --- Introduction --- #
        s = 'Class distribution:\n'
        # ---  Head  --- #
        s += 'CLASS           , PRED. COUNT   , PRED. PERCENT., ' \
            'TRUE COUNT    , TRUE PERCENT. \n'
        # ---  Body  --- #
        yhat_percentage = 100 * self.yhat_count / np.sum(self.yhat_count)
        y_percentage = 100 * self.y_count / np.sum(self.y_count)
        for i in range(len(self.class_names)):
            s += f'{self.class_names[i]:16.16}, ' \
                 f'{self.yhat_count[i]:14}, ' \
                 f'{yhat_percentage[i]:14.3f}, ' \
                 f'{self.y_count[i]:14}, ' \
                 f'{y_percentage[i]:14.3f}\n'
        # Return
        return s

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(
        self,
        report_path,
        class_report_path=None,
        confusion_matrix_report_path=None,
        class_distribution_report_path=None,
        out_prefix=None
    ):
        """
        Write the classification report to a file.

        :param report_path: See :class:`.ClassificationEvaluator`.
        :param class_report_path: See :class:`.ClassificationEvaluator`.
        :param confusion_matrix_report_path: See
            :class:`.ClassificationEvaluator`.
        :param class_distribution_report_path: See
            :class:`.ClassificationEvaluator`.
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Prepare many reports
        report_names = [
            'global evaluation',
            'class evaluation',
            'confusion matrix',
            'class distribution'
        ]
        paths = [
            report_path,
            class_report_path,
            confusion_matrix_report_path,
            class_distribution_report_path
        ]
        checks = [
            self.has_global_eval_info,
            self.has_class_eval_info,
            self.has_confusion_matrix,
            self.has_class_distribution_info
        ]
        to_strings = [
            self.to_global_eval_string,
            self.to_class_eval_string,
            self.to_confusion_matrix_string,
            self.to_class_distribution_string
        ]

        # Do many reports
        for i in range(len(paths)):
            # Extract iteration variables
            report_name = report_names[i]
            path = paths[i]
            check = checks[i]
            to_string = to_strings[i]
            # Check info
            if not check():
                LOGGING.LOGGER.debug(
                    'ClassificationReport did NOT write report on '
                    f'{report_name} to "{path}"'
                )
                continue
            # Expand path if necessary
            if out_prefix is not None and path[0] == "*":
                path = out_prefix[:-1] + path[1:]
            # Check path
            IOUtils.validate_path_to_directory(
                os.path.dirname(path),
                'Cannot find the directory to write the report:'
            )
            # Write
            with open(path, 'w') as outf:
                outf.write(to_string())
            # Log
            LOGGING.LOGGER.info(f'Report on {report_name} written to "{path}"')

    # ---  CHECK METHODS  --- #
    # ----------------------- #
    def has_global_eval_info(self):
        """
        Check whether the report contains information about the global
        evaluation.

        :return: True if the report contains information about the global
            evaluation, False otherwise.
        """
        return self.metric_scores is not None and self.metric_names is not None

    def has_class_eval_info(self):
        """
        Check whether the report contains information about the class-wise
        evaluation.

        :return: True if the report contains information about the class-wise
            evaluation, False otherwise.
        """
        return (
            self.class_names is not None and
            self.class_metric_scores is not None and
            self.class_metric_names is not None
        )

    def has_confusion_matrix(self):
        """
        Check whether the report contains the confusion matrix.

        :return: True if the report contains the confusion matrix, False
            otherwise.
        """
        return self.conf_mat is not None

    def has_class_distribution_info(self):
        """
        Check whether the report contains information about the class
        distribution.

        :return: True if the report contains information about the class
            distribution, False otherwise.
        """
        return (
            self.class_names is not None and
            self.yhat_count is not None and
            self.y_count is not None
        )
