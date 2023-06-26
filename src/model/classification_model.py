# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC

from src.model.model import Model
import src.main.main_logger as LOGGING
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, jaccard_score, matthews_corrcoef, cohen_kappa_score
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassificationModel(Model, ABC):
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing some baseline methods for classification models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type ClassificationModel
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the ClassificationModel
        self.autoval_metrics_names = kwargs.get('autoval_metrics', None)
        if self.autoval_metrics_names is not None:
            self.autoval_metrics = self.autoval_metrics_from_names(
                self.autoval_metrics_names
            )

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def autoval(self, y, yhat, info=True):
        """
        Auto validation during training for classification models.

        :param yhat: The predicted classes.
        :param y: The expected classes.
        :return: The results of the auto validation.
        :rtype: :class:`np.ndarray`
        """
        evals = np.array([
            metric(y, yhat) for metric in self.autoval_metrics
        ])
        if info:
            evals_str = "Classification auto validation:"
            for i, evali in enumerate(evals):
                evals_str += "\n{namei} = {evali:.3f} %".format(
                    namei=self.autoval_metrics_names[i],
                    evali=100*evali
                )
            LOGGING.LOGGER.info(evals_str)
        return evals

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def autoval_metrics_from_names(names):
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

