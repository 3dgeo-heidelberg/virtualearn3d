# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
from src.eval.rand_forest_evaluator import RandForestEvaluator
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import time


# ---   CLASS   --- #
# ----------------- #
class RandomForestClassificationModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    RandomForest model for classification tasks.
    See :class:`.Model`.

    :ivar model_args: The arguments to initialize a new RandomForest model.
    :vartype model_args: dict
    :ivar model: The internal representation of the model.
    :vartype model: :class:`RandomForestClassifier`
    :ivar importance_report_path: Path to the file to store the report.
    :vartype importance_report_path: str
    :ivar importance_report_permutation: Flag to control whether to include
        the permutation importance in the report (True, default) or
        not (False).
    :vartype importance_report_permutation_importance: bool
    :ivar decision_plot_path: Path to the file to store the plots representing
        the decision trees in the random forest. If only one decision tree
        is going to be exported, the path is used literally. Otherwise,
        incrementally updated paths by appending "_n" before the file extension
        will be considered.
    :vartype decision_plot_path: str
    :ivar decision_plot_trees: The number of decision trees to consider. If
        -1, then all the decision trees will be considered.
    :vartype decision_plot_trees: int
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        RandomForestClassificationModel from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            RandomForestClassificationModel
        """
        # Initialize from parent
        kwargs = ClassificationModel.extract_model_args(spec)
        # Extract particular arguments for Random Forest
        kwargs['importance_report_path'] = spec.get(
            'importance_report_path', None
        )
        kwargs['importance_report_permutation'] = spec.get(
            'importance_report_permutation', None
        )
        kwargs['decision_plot_path'] = spec.get('decision_plot_path', None)
        kwargs['decision_plot_trees'] = spec.get('decision_plot_trees', None)
        kwargs['decision_plot_max_depth'] = spec.get(
            'decision_plot_max_depth', None
        )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of RandomForestModel.

        :param kwargs: The attributes for the RandomForestClassificationModel
            that will also be passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the RandomForestClassificationModel
        self.model = None
        self.importance_report_path = kwargs.get(
            'importance_report_path', None
        )
        self.importance_report_permutation = kwargs.get(
            'importance_report_permutation', True
        )
        self.decision_plot_path = kwargs.get('decision_plot_path', None)
        self.decision_plot_trees = kwargs.get('decision_plot_trees', 0)
        self.decision_plot_max_depth = kwargs.get('decision_plot_max_depth', 5)

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def prepare_model(self):
        """
        Prepare a random forest classifier with current model arguments

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        """
        if self.model_args is not None:
            self.model = RandomForestClassifier(**self.model_args)
        else:
            LOGGING.LOGGER.info(
                "Preparing RandomForestClassificationModel with no "
                "`model_args`"
            )
            self.model = RandomForestClassifier()
        return self.model

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y, info=True):
        """
        The fundamental training logic to train a random forest classifier.

        See :class:`.ClassificationModel` and :class:`.Model`.
        Also see :meth:`model.Model.training`.
        """
        # Initialize model instance
        self.prepare_model()
        # Train the model
        start = time.perf_counter()
        self.model = self.model.fit(X, y)
        end = time.perf_counter()
        # Log end of execution
        if info:
            LOGGING.LOGGER.info(
                'RandomForestClassificationModel trained in '
                f'{end-start:.3f} seconds.'
            )

    def on_training_finished(self, X, y):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Report feature importances and plot decision trees
        start = time.perf_counter()
        ev = RandForestEvaluator(
            problem_name='Trained Random Forest Classification Model',
            num_decision_trees=self.decision_plot_trees,
            compute_permutation_importance=self.importance_report_permutation,
            max_tree_depth=self.decision_plot_max_depth
        ).eval(self, X=X, y=y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'RandomForestClassificationModel computed "on training finished" '
            f'evaluation in {end-start:.3f} seconds.'
        )
        if ev.can_report():
            report = ev.report()
            LOGGING.LOGGER.info(report.to_string())
            if self.importance_report_path is not None:
                report.to_file(path=self.importance_report_path)
        else:
            LOGGING.LOGGER.warning(
                'RandomForestClassificationModel could not report.'
            )
        if self.decision_plot_path is not None:
            if ev.can_plot():
                ev.plot(path=self.decision_plot_path).plot()
            else:
                LOGGING.LOGGER.warning(
                    'RandomForestClassificationModel could not plot.'
                )

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, zout=None):
        """
        See :meth:`model.Model._predict`.
        """
        if zout is not None:
            zout.append(self.model.predict_proba(X))
        return self.model.predict(X)
