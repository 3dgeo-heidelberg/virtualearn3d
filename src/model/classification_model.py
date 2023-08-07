# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.model.model import Model
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.eval.classification_evaluator import ClassificationEvaluator
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassificationModel(Model, ABC):
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing some baseline methods for classification models.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a ClassificationModel
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a ClassificationModel.
        """
        # Initialize from parent
        kwargs = Model.extract_model_args(spec)
        # Extract particular arguments for classification models
        kwargs['autoval_metrics'] = spec.get('autoval_metrics', None)
        kwargs['class_names'] = spec.get('class_names', None)
        if kwargs['class_names'] is None:
            model_args = spec.get('model_args', None)
            if model_args is not None:
                kwargs['class_names'] = model_args.get('class_names', None)
        kwargs['training_evaluation_metrics'] = spec.get(
            'training_evaluation_metrics', None
        )
        kwargs['training_class_evaluation_metrics'] = spec.get(
            'training_class_evaluation_metrics', None
        )
        kwargs['training_evaluation_report_path'] = spec.get(
            'training_evaluation_report_path', None
        )
        kwargs['training_class_evaluation_report_path'] = spec.get(
            'training_class_evaluation_report_path', None
        )
        kwargs['training_confusion_matrix_report_path'] = spec.get(
            'training_confusion_matrix_report_path', None
        )
        kwargs['training_confusion_matrix_plot_path'] = spec.get(
            'training_confusion_matrix_plot_path', None
        )
        kwargs['training_class_distribution_report_path'] = spec.get(
            'training_class_distribution_report_path', None
        )
        kwargs['training_class_distribution_plot_path'] = spec.get(
            'training_class_distribution_plot_path', None
        )
        kwargs['training_classified_point_cloud_path'] = spec.get(
            'training_classified_point_cloud_path', None
        )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type ClassificationModel
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the ClassificationModel
        if self.autoval_metrics_names is not None:
            self.autoval_metrics = self.autoval_metrics_from_names(
                self.autoval_metrics_names
            )
        self.class_names = kwargs.get('class_names', None)
        # Training evaluation attributes (do not confuse with autoval or kfold)
        self.training_evaluation_metrics = kwargs.get(
            'training_evaluation_metrics', None
        )
        self.training_class_evaluation_metrics = kwargs.get(
            'training_class_evaluation_metrics', None
        )
        self.training_evaluation_report_path = kwargs.get(
            'training_evaluation_report_path', None
        )
        self.training_class_evaluation_report_path = kwargs.get(
            'training_class_evaluation_report_path', None
        )
        self.training_confusion_matrix_report_path = kwargs.get(
            'training_confusion_matrix_report_path', None
        )
        self.training_confusion_matrix_plot_path = kwargs.get(
            'training_confusion_matrix_plot_path', None
        )
        self.training_class_distribution_report_path = kwargs.get(
            'training_class_distribution_report_path', None
        )
        self.training_class_distribution_plot_path = kwargs.get(
            'training_class_distribution_plot_path', None
        )
        self.training_classified_point_cloud_path = kwargs.get(
            'training_classified_point_cloud_path', None
        )

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def autoval(self, y, yhat, info=True):
        """
        Auto validation during training for classification models.

        :param yhat: The predicted classes.
        :param y: The expected classes.
        :param info: True to log an info message with the auto validation,
            False otherwise.
        :return: The results of the auto validation.
        :rtype: :class:`np.ndarray`
        """
        # TODO Rethink : Implement through Evaluator ?
        evals = np.array([
            metric(y, yhat) for metric in self.autoval_metrics
        ])
        if info:
            evals_str = "Classification auto validation:"
            for i, evali in enumerate(evals):
                evals_str += "\n{namei:8.8} = {evali:.3f}%".format(
                    namei=self.autoval_metrics_names[i],
                    evali=100*evali
                )
            LOGGING.LOGGER.info(evals_str)
        return evals

    def on_training_finished(self, X, y, yhat=None):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Compute the estimations on the training dataset if not given
        if yhat is None:
            yhat = self._predict(X, F=None)
        # Training evaluation
        training_eval = ClassificationEvaluator(
            class_names=self.class_names,
            metrics=self.training_evaluation_metrics,
            class_metrics=self.training_class_evaluation_metrics,
            report_path=self.training_evaluation_report_path,
            class_report_path=self.training_class_evaluation_report_path,
            confusion_matrix_report_path=self.training_confusion_matrix_report_path,
            confusion_matrix_plot_path=self.training_confusion_matrix_plot_path,
            class_distribution_report_path=self.training_class_distribution_report_path,
            class_distribution_plot_path=self.training_class_distribution_plot_path
        ).eval(yhat, y=y)
        if training_eval.can_report():
            training_eval_report = training_eval.report()
            LOGGING.LOGGER.info(
                f'{self.__class__.__name__} training evaluation:\n'
                f'{training_eval_report.to_string()}'
            )
            training_eval_report.to_file(
                self.training_evaluation_report_path,
                class_report_path=self.training_class_evaluation_report_path,
                confusion_matrix_report_path=self.training_confusion_matrix_report_path,
                class_distribution_report_path=self.training_class_distribution_report_path
            )
        if training_eval.can_plot():
            training_eval.plot(
                path=self.training_confusion_matrix_plot_path,
                class_distribution_path=self.training_class_distribution_plot_path
            ).plot()

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def autoval_metrics_from_names(names):
        """
        See
        :meth:`classification_evaluator.ClassificationEvaluator.metrics_from_names`
        .
        """
        return ClassificationEvaluator.metrics_from_names(names)

    # ---  PICKLE METHODS  --- #
    # ------------------------ #
    def __getstate__(self):
        """
        Method to be called when saving the serialized classification model.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        state = self.__dict__.copy()
        # Remove metrics (because they use lambda functions)
        if state.get('autoval_metrics', None) is not None:
            del state['autoval_metrics']
        # Return state dictionary
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized classification model.

        :param state: The state's dictionary of the saved classification model.
        :return: Nothing, but modifies the internal state of the object.
        """
        # Defualt update
        self.__dict__.update(state)
        # Recompute lambda functions
        if self.autoval_metrics_names is not None:
            self.autoval_metrics = self.autoval_metrics_from_names(
                self.autoval_metrics_names
            )
