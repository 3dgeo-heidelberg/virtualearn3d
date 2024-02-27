# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.classification_evaluation import ClassificationEvaluation
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, jaccard_score, matthews_corrcoef, cohen_kappa_score
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ClassificationEvaluator(Evaluator):
    r"""
    :author: Alberto M. Esmoris Pena

    Class to evaluate classification-like predictions against
    expected/reference classes.

    :ivar metrics: The name of the metrics for overall evaluation.
    :vartype metrics: list
    :ivar class_metrics: The name of the metrics for the class-wise evaluation.
    :vartype class_metrics: list
    :ivar class_names: The name for each class.
    :vartype class_names: list
    :ivar metricf: The functions to compute each metric. The metric :math:`j`
        will be computed as :math:`f_j(y, \hat{y})`.
    :vartype metricf: list
    :ivar class_metricf: The function to compute class-wise metrics. The metric
        :math:`j` for class :math:`i` will be computed as
        :math:`f_j(y, \hat{y}, i)`.
    :vartype class_metricf: list
    :ivar report_path: The path to write the global evaluation report.
    :vartype report_path: str
    :ivar class_report_path: The path to write the class-wise evaluation
        report.
    :vartype class_report_path: str
    :ivar confusion_matrix_report_path: The path to write the confusion
        matrix report.
    :vartype confusion_matrix_report_path: str
    :ivar confusion_matrix_plot_path: The path to write the plot representing
        the confusion matrix.
    :vartype confusion_matrix_plot_path: str
    :ivar class_distribution_report_path: The path to write the class
        distribution report.
    :vartype class_distribution_report_path: str
    :ivar class_distribution_plot_path: the path to write the plot representing
        the class distribution.
    :vartype class_distribution_plot_path: str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_eval_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        ClassificationEvaluator from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            ClassificationEvaluator.
        """
        # Initialize
        kwargs = {
            'class_names': spec.get('class_names', None),
            'ignore_classes': spec.get('ignore_classes', None),
            'metrics': spec.get('metrics', None),
            'class_metrics': spec.get('class_metrics', None),
            'report_path': spec.get('report_path', None),
            'class_report_path': spec.get('class_report_path', None),
            'confusion_matrix_report_path': spec.get(
                'confusion_matrix_report_path', None
            ),
            'confusion_matrix_plot_path': spec.get(
                'confusion_matrix_plot_path', None
            ),
            'class_distribution_report_path': spec.get(
                'class_distribution_report_path', None
            ),
            'class_distribution_plot_path': spec.get(
                'class_distribution_plot_path', None
            )
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassificationEvaluator.

        :param kwargs: The attributes for the ClassificationEvaluator.
        """
        # Call parent's init
        kwargs['problem_name'] = 'CLASSIFICATION'
        super().__init__(**kwargs)
        # Assign ClassificationEvaluator attributes
        self.metrics = kwargs.get('metrics', None)
        self.class_metrics = kwargs.get('class_metrics', None)
        self.class_names = kwargs.get('class_names', None)
        self.ignore_classes = kwargs.get('ignore_classes', None)  # TODO Rethink : Doc
        self.metricf = None
        if self.metrics is not None:
            self.metricf = ClassificationEvaluator.metrics_from_names(
                self.metrics
            )
        self.class_metricf = None
        if self.class_metrics:
            self.class_metricf = \
                ClassificationEvaluator.class_metrics_from_names(
                    self.class_metrics
                )
        self.report_path = kwargs.get('report_path', None)
        self.class_report_path = kwargs.get('class_report_path', None)
        self.confusion_matrix_report_path = kwargs.get(
            'confusion_matrix_report_path', None
        )
        self.confusion_matrix_plot_path = kwargs.get(
            'confusion_matrix_plot_path', None
        )
        self.class_distribution_report_path = kwargs.get(
            'class_distribution_report_path', None
        )
        self.class_distribution_plot_path = kwargs.get(
            'class_distribution_plot_path', None
        )

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, yhat, y=None, **kwargs):
        r"""
        Evaluate predicted classes (:math:`\hat{y}`) against expected/reference
        classes (:math:`y`).

        :param yhat: The predictions to be evaluated.
        :param y: The expected/reference values to evaluate the predictions
            against.
        :return: The evaluation of the classification.
        :rtype: :class:`.ClassificationEvaluation`
        """
        start = time.perf_counter()
        # Validate arguments
        if y is None:
            raise EvaluatorException(
                'ClassificationEvaluator cannot evaluate without the expected '
                'or reference labels.'
            )
        # Determine classes (names and numbers)
        if self.class_names is None:  # Automatically determine class_names
            class_nums = np.unique(np.concatenate([yhat, y]))
            class_names = [f'C{i}' for i in class_nums]
        else:  # Determine class_nums from given class_names
            class_nums = np.array(
                [i for i in range(len(self.class_names))],
                dtype=int
            )
            class_names = self.class_names
        if self.ignore_classes is not None:
            ignore_classes_indices = \
                ClassificationEvaluator.get_indices_from_names(
                    class_names, self.ignore_classes
                )
            # TODO Rethink : Remove code below iff not necessary
            #class_names = [
            #    name for name in class_names if name not in self.ignore_classes
            #]
            ignore_mask = ClassificationEvaluator.find_ignore_mask(
                y, ignore_classes_indices
            )
            y = ClassificationEvaluator.remove_indices(y, ignore_mask)
            yhat = ClassificationEvaluator.remove_indices(yhat, ignore_mask)
        # Evaluate : class distribution
        yhat_count, yhat_bin = np.histogram(
            yhat,
            bins=np.linspace(0, len(class_nums)-1, len(class_nums)+1)
        )
        y_count, y_bin = np.histogram(
            y, bins=len(class_nums), range=(0, len(class_nums))
        )
        # Evaluate : confusion matrix
        conf_mat = confusion_matrix(y, yhat)
        # Evaluate : metrics
        scores = np.array([
            self.metricf[i](y, yhat) for i in range(len(self.metricf))
        ])
        # Evaluate : class-wise metrics
        class_scores = np.array([  # [i][j] is metric i on class j
            [
                self.class_metricf[i](y, yhat, j)
                for j in range(len(class_names))
            ]
            for i in range(len(self.class_metricf))
        ])
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClassificationEvaluator evaluated {len(yhat)} points in '
            f'{end-start:.3f} seconds.'
        )
        # Return
        return ClassificationEvaluation(
            class_names=class_names,
            yhat_count=yhat_count,
            y_count=y_count,
            conf_mat=conf_mat,
            metric_names=self.metrics,
            metric_scores=scores,
            class_metric_names=self.class_metrics,
            class_metric_scores=class_scores
        )

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
            report_path = kwargs.get(
                'report_path', self.report_path
            )
            class_report_path = kwargs.get(
                'class_report_path', self.class_report_path
            )
            confusion_matrix_report_path = kwargs.get(
                'confusion_matrix_report_path',
                self.confusion_matrix_report_path
            )
            class_distrib_report_path = kwargs.get(
                'class_distribution_report_path',
                self.class_distribution_report_path
            )
            if(
                report_path is not None or
                class_report_path is not None or
                confusion_matrix_report_path is not None or
                class_distrib_report_path is not None
            ):
                start = time.perf_counter()
                report.to_file(
                    report_path,
                    class_report_path=class_report_path,
                    confusion_matrix_report_path=confusion_matrix_report_path,
                    class_distribution_report_path=class_distrib_report_path,
                    out_prefix=out_prefix
                )
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    f'Classification reports written in {end-start:.3f} '
                    'seconds.'
                )
        if ev.can_plot():
            plot_path = kwargs.get('plot_path', kwargs.get(
                'confusion_matrix_plot_path',
                self.confusion_matrix_plot_path
            ))
            class_distribution_plot_path = kwargs.get(
                'class_distribution_plot_path',
                self.class_distribution_plot_path
            )
            if plot_path is not None:
                start = time.perf_counter()
                ev.plot(
                    path=plot_path,
                    class_distribution_path=class_distribution_plot_path
                ).plot(out_prefix=out_prefix)
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    f'Classification plots written in {end-start:.3f} '
                    'seconds.'
                )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def metrics_from_names(names):
        """
        Obtain a list of metrics that can be evaluated for vectors of classes
        y (expected), and yhat (predicted).

        :param names: The names of the metrics. Currently supported metrics are
            Overall Accuracy "OA", Precision "P", Recall "R", F1 score "F1",
            Intersection over Union "IoU", weighted Precision "wP", weighted
            Recall "wR", weighted F1 score "wF1", weighted Intersection over
            Union "wIoU", Matthews Correlation Coefficient "MCC", and Kohen's
            Kappa score "Kappa".
        :return: List of metrics such that metric_i(y, yhat) can be invoked.
        :rtype: list
        """
        metrics = []
        if "OA" in names:
            metrics.append(accuracy_score)
        if "P" in names:
            metrics.append(
                lambda y, yhat: precision_score(
                    y, yhat, average='macro'
                )
            )
        if "R" in names:
            metrics.append(
                lambda y, yhat: recall_score(
                    y, yhat, average='macro'
                )
            )
        if "F1" in names:
            metrics.append(
                lambda y, yhat: f1_score(
                    y, yhat, average='macro'
                )
            )
        if "IoU" in names:
            metrics.append(
                lambda y, yhat: jaccard_score(
                    y, yhat, average='macro'
                )
            )
        if "wP" in names:
            metrics.append(
                lambda y, yhat: precision_score(
                    y, yhat, average='weighted'
                )
            )
        if "wR" in names:
            metrics.append(
                lambda y, yhat: recall_score(
                    y, yhat, average='weighted'
                )
            )
        if "wF1" in names:
            metrics.append(
                lambda y, yhat : f1_score(
                    y, yhat, average='weighted'
                )
            )
        if "wIoU" in names:
            metrics.append(
                lambda y, yhat : jaccard_score(
                    y, yhat, average='weighted'
                )
            )
        if "MCC" in names:
            metrics.append(matthews_corrcoef)
        if "Kappa" in names:
            metrics.append(cohen_kappa_score)
        return metrics

    @staticmethod
    def class_metrics_from_names(names):
        """
        Obtain a list of class-wise metrics that can be evaluated for vectors
        of classes :math:`y` (expected), and :math:`yhat` (predicted).

        :param names: The names of the class-wise metrics. Currently supported
            Precision "P", Recall "R", F1 score "F1",
            and Intersection over Union "IoU".
        :return: List of metrics such that metric_i(y, yhat) can be invoked.
        :rtype: list
        """
        metrics = []
        if "P" in names:
            metrics.append(
                lambda y, yhat, i: precision_score(
                    y, yhat, average='macro', labels=[i]
                )
            )
        if "R" in names:
            metrics.append(
                lambda y, yhat, i: recall_score(
                    y, yhat, average='macro', labels=[i]
                )
            )
        if "F1" in names:
            metrics.append(
                lambda y, yhat, i: f1_score(
                    y, yhat, average='macro', labels=[i]
                )
            )
        if "IoU" in names:
            metrics.append(
                lambda y, yhat, i: jaccard_score(
                    y, yhat, average='macro', labels=[i]
                )
            )
        if "Kappa" in names:
            metrics.append(
                lambda y, yhat, i: cohen_kappa_score(
                    y, yhat, labels=[i]
                )
            )
        return metrics

    @staticmethod
    def get_indices_from_names(all_names, target_names):
        """
        Find the indices of the target names with respect to all names.

        :param all_names: The list of all names.
        :type all_names: list
        :param target_names: The list of target names.
        :type target_names: list
        :return: The indices of `target_names` in the `all_names` list
        :rtype: :class:`np.ndarray` of int
        """
        indices = []
        for target_name in target_names:
            for idx, name in enumerate(all_names):
                if target_name == name:
                    indices.append(idx)
                    break
        return np.array(indices, dtype=int)

    @staticmethod
    def find_ignore_mask(y, indices):
        """
        Find the boolean ignore mask (True to ignore, False otherwise).

        For any point whose class (given by `y`) matches any target index
        (given by `indices`) a true must be stored in the corresponding
        element of the boolean mask.

        :param y: A vector of point-wise classes.
        :type y: :class:`np.ndarray`
        :param indices: The class indices to search for.
        :type indices: :class:`np.ndarray`
        :return: The boolean mask specifying what points must be ignored.
        :rtype: :class:`np.ndarray` of bool
        """
        mask = np.zeros(y.shape, dtype=bool)
        for index in indices:
            mask = mask + (y == index)
        return mask

    @staticmethod
    def remove_indices(y, mask):
        """
        Preserve only those points for which the boolean mask is False (True
        means it must be ignored).

        :param y: The points to be filtered by the mask.
        :param mask: The mask defining the removal filter.
        :return: The input points without the ignored ones.
        """
        return y[~mask]
