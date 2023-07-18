# ---   IMPORTS  --- #
# ------------------ #
from src.model.model import Model
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
from src.main.vl3d_exception import VL3DException
from enum import Enum


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ModelOpException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to model operations.
    See :class:`.VL3DException`
    """
    pass


# ---   MODEL   --- #
# ----------------- #
class ModelOp:
    """
    :author: Alberto M. Esmoris Pena

    Class wrapping a model associated to an operation (useful to handle
    different model operations like train and predict during pipelines).

    :ivar model: The model.
    :vartype model: :class:`.Model`
    :ivar op: The operation.
    :vartype op: enum
    """

    # The enumeration
    OP = Enum(
        'OP',
        ['TRAIN', 'PREDICT'],
        module=__name__,
        qualname='ModelOp.OP'
    )

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, model, op):
        """
        Root initialization for any instance of type ModelOp.

        :param model: The model associated to the operation.
        :param op: The operation associated to the model.
        """
        # Assign attributes
        self.model = model
        self.op = op
        # Validate attributes
        if self.model is None:
            raise ModelOpException(
                'Model operation without model is not supported.'
            )
        if not isinstance(model, Model):
            raise ModelOpException(
                'Model operation is not supported for model objects that do '
                'not extend from the Model class.'
            )
        if op is None:
            raise ModelOpException(
                'Model operation without operation is not supported.'
            )
        if not isinstance(op, ModelOp.OP):
            raise ModelOpException(
                'Model operation requires the operation is specified using '
                'the ModelOp.OP enumeration.'
            )

    # ---   CALL   --- #
    # ---------------- #
    def __call__(self, pcloud=None, out_prefix=None):
        """
        Make the model do the operation.

        :param pcloud: Input point cloud. Train and predict operations require
            it is not None.
        :type pcloud: :class:`.PointCloud` or None
        :param out_prefix: Optional argument to specify the output prefix for
            any path in the model that starts with '*'.
        :type param: str or None
        :return: Whatever the operation returns.
        """
        if self.op == ModelOp.OP.TRAIN:
            if pcloud is None:
                raise ModelOpException(
                    'Train as model operation requires an input point cloud. '
                    'None was given.'
                )
            old_paths = self.update_model_paths(out_prefix)
            model = self.model.train(pcloud)
            self.restore_model_paths(old_paths)
            return model
        if self.op == ModelOp.OP.PREDICT:
            if pcloud is None:
                raise ModelOpException(
                    'Predict as model operation requires an input point '
                    'cloud. None was given.'
                )
            old_paths = self.update_model_paths(out_prefix)
            predictions = self.model.predict(pcloud)
            self.restore_model_paths(old_paths)
            return predictions

    # ---  PATH HANDLING  --- #
    # ----------------------- #
    def update_model_paths(self, out_prefix):
        """
        Update the output paths in the model object.

        For each path that starts with "*" it will be updated to replace the
        "*" by the prefix.

        :param out_prefix: The prefix for the output paths. It is expected to
            end with "*".
        :type out_prefix: str
        :return: A dictionary with the paths before the update. Note the model
            is updated in place.
        :rtype: dict
        """
        # Nothing to do if there is no output prefix
        if out_prefix is None:
            return None
        # Initialize dictionary of old paths
        old_paths = {}
        # Handle model paths
        stratkfold_plot_path = self.model.stratkfold_plot_path
        old_paths['stratkfold_plot_path'] = stratkfold_plot_path
        if ModelOp.path_needs_update(stratkfold_plot_path):
            self.model.stratkfold_plot_path = ModelOp.merge_path(
                out_prefix, stratkfold_plot_path
            )
        stratkfold_report_path = self.model.stratkfold_report_path
        old_paths['stratkfold_report_path'] = stratkfold_report_path
        if ModelOp.path_needs_update(stratkfold_report_path):
            self.model.stratkfold_report_path = ModelOp.merge_path(
                out_prefix, stratkfold_report_path
            )
        # Handle model's hypertuner paths
        if self.model.hypertuner is not None:
            hypertuner_report_path = self.model.hypertuner.report_path
            old_paths['hypertuner_report_path'] = hypertuner_report_path
            if ModelOp.path_needs_update(hypertuner_report_path):
                self.model.hypertuner.report_path = ModelOp.merge_path(
                    out_prefix, hypertuner_report_path
                )
        # Handle RandomForest paths
        if isinstance(self.model, RandomForestClassificationModel):
            importance_report_path = self.model.importance_report_path
            old_paths['importance_report_path'] = importance_report_path
            if ModelOp.path_needs_update(importance_report_path):
                self.model.importance_report_path = ModelOp.merge_path(
                    out_prefix, importance_report_path
                )
            decision_plot_path = self.model.decision_plot_path
            old_paths['decision_plot_path'] = decision_plot_path
            if ModelOp.path_needs_update(decision_plot_path):
                self.model.decision_plot_path = ModelOp.merge_path(
                    out_prefix, decision_plot_path
                )
        # Return dictionary to reverse updates
        return old_paths

    def restore_model_paths(self, old_paths):
        """
        Restore previously updated model paths to their original values.

        :param old_paths: The dictionary with the paths before the update.
        :return: Nothing, the model is restored in place.
        """
        # Nothing to do if there is nothing to restore
        if old_paths is None:
            return
        # Restore model paths
        self.model.stratkfold_plot_path = old_paths.get(
            'stratkfold_plot_path'
        )
        self.model.stratkfold_report_path = old_paths.get(
            'stratkfold_report_path'
        )
        # Restore model's hypertuner paths
        if self.model.hypertuner is not None:
            self.model.hypertuner.report_path = old_paths.get(
                'hypertuner_report_path'
            )
        # Restore random forest paths
        if isinstance(self.model, RandomForestClassificationModel):
            self.model.importance_report_path = old_paths.get(
                'importance_report_path'
            )
            self.model.decision_plot_path = old_paths.get(
                'decision_plot_path'
            )

    @staticmethod
    def path_needs_update(path : str) -> bool:
        """
        Check whether a model path needs to be updated or not.

        A model path needs to be updated if it starts by "*".

        :param path: The model path to be checked.
        :return: True if the model path needs to be updated, False otherwise.
        :rtype: bool
        """
        return path is not None and path[0] == '*'

    @staticmethod
    def merge_path(out_prefix : str, path : str) -> str:
        """
        Merge the output prefix to the model path assuming the output prefix
        ends by "*" and the model path starts with "*".

        :param out_prefix: The output prefix for the merge.
        :param path: The model path for the merge.
        :return: The merged path.
        :rtype: str
        """
        return out_prefix[:-1] + path[1:]
