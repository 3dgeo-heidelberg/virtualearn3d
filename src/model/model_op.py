# ---   IMPORTS  --- #
# ------------------ #
from src.model.model import Model
from src.model.classification_model import ClassificationModel
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
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
        # Handle ClassificationModel paths
        if isinstance(self.model, ClassificationModel):
            training_evaluation_report_path = \
                self.model.training_evaluation_report_path
            old_paths['training_evaluation_report_path'] = \
                self.model.training_evaluation_report_path
            if ModelOp.path_needs_update(training_evaluation_report_path):
                self.model.training_evaluation_report_path = \
                    ModelOp.merge_path(
                        out_prefix, training_evaluation_report_path
                    )
            training_class_evaluation_report_path = \
                self.model.training_class_evaluation_report_path
            old_paths['training_class_evaluation_report_path'] = \
                self.model.training_class_evaluation_report_path
            if ModelOp.path_needs_update(
                training_class_evaluation_report_path
            ):
                self.model.training_class_evaluation_report_path = \
                    ModelOp.merge_path(
                        out_prefix, training_class_evaluation_report_path
                    )
            training_confusion_matrix_report_path = \
                self.model.training_confusion_matrix_report_path
            old_paths['training_confusion_matrix_report_path'] = \
                self.model.training_confusion_matrix_report_path
            if ModelOp.path_needs_update(
                training_confusion_matrix_report_path
            ):
                self.model.training_confusion_matrix_report_path = \
                    ModelOp.merge_path(
                        out_prefix, training_confusion_matrix_report_path
                    )
            training_confusion_matrix_plot_path = \
                self.model.training_confusion_matrix_plot_path
            old_paths['training_confusion_matrix_plot_path'] = \
                self.model.training_confusion_matrix_plot_path
            if ModelOp.path_needs_update(
                training_confusion_matrix_plot_path
            ):
                self.model.training_confusion_matrix_plot_path = \
                    ModelOp.merge_path(
                        out_prefix, training_confusion_matrix_plot_path
                    )
            training_class_distribution_report_path = \
                self.model.training_class_distribution_report_path
            if ModelOp.path_needs_update(
                training_class_distribution_report_path
            ):
                self.model.training_class_distribution_report_path = \
                    ModelOp.merge_path(
                        out_prefix, training_class_distribution_report_path
                    )
            training_class_distribution_plot_path = \
                self.model.training_class_distribution_plot_path
            if ModelOp.path_needs_update(
                training_class_distribution_plot_path
            ):
                self.model.training_class_distribution_plot_path = \
                    ModelOp.merge_path(
                        out_prefix, training_class_distribution_plot_path
                    )
            training_classified_point_cloud_path = \
                self.model.training_classified_point_cloud_path
            if ModelOp.path_needs_update(
                training_classified_point_cloud_path
            ):
                self.model.training_classified_point_cloud_path = \
                    ModelOp.merge_path(
                        out_prefix, training_classified_point_cloud_path
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
        # Handle deep learning paths
        if self.model.is_deep_learning_model():
            self.update_dl_model_paths(out_prefix, old_paths)
        # Return dictionary to reverse updates
        return old_paths

    def update_dl_model_paths(self, out_prefix, old_paths):
        """
        Update the model assuming it is a deep learning model. This method
        will not check that self.model is a deep learning model. Instead,
        it assumes it is because otherwise the method MUST NOT be called.

        See :meth:`model_op.ModelOp.update_model_paths`.

        :param old_paths: The output parameter. It must be a dictionary where
            any modified path (value) will be stored before it is updated,
            associated to its variable (key).
        :type old_paths: dict
        :return: Nothing at all, the old_paths list argument is updated if
            necessary.
        """
        # Extract objects of interest
        model_args = self.model.model_args
        model_handling = model_args.get('model_handling', None)
        preproc = model_args.get('pre_processing', None)
        # Handle training activations path
        training_activations_path = self.model.training_activations_path
        if ModelOp.path_needs_update(training_activations_path):
            self.model.training_activations_path = ModelOp.merge_path(
                out_prefix, training_activations_path
            )
        old_paths['training_activations_path'] = training_activations_path
        # Handle architecture graph path
        architecture_graph_path = model_args.get(
            'architecture_graph_path', None
        )
        if ModelOp.path_needs_update(architecture_graph_path):
            model_args['architecture_graph_path'] = ModelOp.merge_path(
                out_prefix, architecture_graph_path
            )
        old_paths['architecture_graph_path'] = architecture_graph_path
        # Handle summary report path
        summary_report_path = model_handling.get('summary_report_path', None)
        if ModelOp.path_needs_update(summary_report_path):
            model_handling['summary_report_path'] = ModelOp.merge_path(
                out_prefix, summary_report_path
            )
        old_paths['summary_report_path'] = summary_report_path
        # Handle training history dir
        training_history_dir = model_handling.get('training_history_dir', None)
        if ModelOp.path_needs_update(training_history_dir):
            model_handling['training_history_dir'] = ModelOp.merge_path(
                out_prefix, training_history_dir
            )
        old_paths['training_history_dir'] = training_history_dir
        # Handle features structuring representation dir
        feat_struct_repr_dir = model_handling.get(
            'features_structuring_representation_dir', None
        )
        if ModelOp.path_needs_update(feat_struct_repr_dir):
            model_handling['features_structuring_representation_dir'] = \
                ModelOp.merge_path(out_prefix, feat_struct_repr_dir)
        old_paths['feat_struct_repr_dir'] = feat_struct_repr_dir
        # Handle rbf feature extraction representation dir
        rbf_feat_extract_repr_dir = model_handling.get(
            'rbf_feature_extraction_representation_dir', None
        )
        if ModelOp.path_needs_update(rbf_feat_extract_repr_dir):
            model_handling['rbf_feature_extraction_representation_dir'] = \
                ModelOp.merge_path(out_prefix, rbf_feat_extract_repr_dir)
        old_paths['rbf_feat_extract_repr_dir'] = rbf_feat_extract_repr_dir
        # Handle rbf feature processing representation dir
        rbf_feat_processing_repr_dir = model_handling.get(
            'rbf_feature_processing_representation_dir', None
        )
        if ModelOp.path_needs_update(rbf_feat_processing_repr_dir):
            model_handling['rbf_feature_processing_representation_dir'] = \
                ModelOp.merge_path(out_prefix, rbf_feat_processing_repr_dir)
        old_paths['rbf_feat_processing_repr_dir'] = rbf_feat_processing_repr_dir
        # Handle KPConv representation dir
        kpconv_representation_dir = model_handling.get(
            'kpconv_representation_dir', None
        )
        if ModelOp.path_needs_update(kpconv_representation_dir):
            model_handling['kpconv_representation_dir'] = \
                ModelOp.merge_path(out_prefix, kpconv_representation_dir)
        old_paths['kpconv_representation_dir'] = kpconv_representation_dir
        # Handle SKPConv representation dir
        skpconv_representation_dir = model_handling.get(
            'skpconv_representation_dir', None
        )
        if ModelOp.path_needs_update(skpconv_representation_dir):
            model_handling['skpconv_representation_dir'] = \
                ModelOp.merge_path(out_prefix, skpconv_representation_dir)
        old_paths['skpconv_representation_dir'] = skpconv_representation_dir
        # Handle checkpoint path
        checkpoint_path = model_handling.get('checkpoint_path', None)
        if ModelOp.path_needs_update(checkpoint_path):
            model_handling['checkpoint_path'] = ModelOp.merge_path(
                out_prefix, checkpoint_path
            )
        old_paths['checkpoint_path'] = checkpoint_path
        # Handle training receptive fields distribution report path
        trf_dist_report = preproc.get(
            'training_receptive_fields_distribution_report_path', None
        )
        if ModelOp.path_needs_update(trf_dist_report):
            preproc['training_receptive_fields_distribution_report_path'] =\
                ModelOp.merge_path(out_prefix, trf_dist_report)
        old_paths['training_receptive_fields_distribution_report_path'] =\
            trf_dist_report
        # Handle training receptive fields distribution plot path
        trf_dist_plot = preproc.get(
            'training_receptive_fields_distribution_plot_path', None
        )
        if ModelOp.path_needs_update(trf_dist_plot):
            preproc['training_receptive_fields_distribution_plot_path'] = \
                ModelOp.merge_path(out_prefix, trf_dist_plot)
        old_paths['training_receptive_fields_distribution_plot_path'] = \
            trf_dist_plot
        # Handle training receptive fields dir
        trf_dir = preproc.get('training_receptive_fields_dir', None)
        if ModelOp.path_needs_update(trf_dir):
            preproc['training_receptive_fields_dir'] = \
                ModelOp.merge_path(out_prefix, trf_dir)
        old_paths['training_receptive_fields_dir'] = trf_dir
        # Handle receptive fields distribution report path
        rf_dist_report = preproc.get(
            'receptive_fields_distribution_report_path', None
        )
        if ModelOp.path_needs_update(rf_dist_report):
            preproc['receptive_fields_distribution_report_path'] = \
                ModelOp.merge_path(out_prefix, rf_dist_report)
        old_paths['receptive_fields_distribution_report_path'] = \
            rf_dist_report
        # Handle receptive fields distribution plot path
        rf_dist_plot = preproc.get(
            'receptive_fields_distribution_plot_path', None
        )
        if ModelOp.path_needs_update(rf_dist_plot):
            preproc['receptive_fields_distribution_plot_path'] = \
                ModelOp.merge_path(out_prefix, rf_dist_plot)
        old_paths['receptive_fields_distribution_plot_path'] = \
            rf_dist_plot
        # Handle receptive fields dir
        rf_dir = preproc.get('receptive_fields_dir', None)
        if ModelOp.path_needs_update(rf_dir):
            preproc['receptive_fields_dir'] = \
                ModelOp.merge_path(out_prefix, rf_dir)
        old_paths['receptive_fields_dir'] = rf_dir
        # Handle training support points report path
        tsp_report = preproc.get('training_support_points_report_path', None)
        if ModelOp.path_needs_update(tsp_report):
            preproc['training_support_points_report_path'] =\
                ModelOp.merge_path(out_prefix, tsp_report)
        old_paths['training_support_points_report_path'] = tsp_report
        # Handle support points report path
        sp_report = preproc.get('support_points_report_path', None)
        if ModelOp.path_needs_update(sp_report):
            preproc['support_points_report_path'] = \
                ModelOp.merge_path(out_prefix, sp_report)
        old_paths['support_points_report_path'] = sp_report
        # Make the changes effective on the arguments
        if model_args is not None:
            self.model.model_args = model_args
        if model_handling is not None:
            self.model.model_args['model_handling'] = model_handling
        if preproc is not None:
            self.model.model_args['pre_processing'] = preproc
        # Make the changes effective on the path-related attributes
        if hasattr(self.model, "update_paths"):
            self.model.update_paths()

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
        # Restore classification model paths
        if isinstance(self.model, ClassificationModel):
            self.model.training_evaluation_report_path = old_paths.get(
                'training_evaluation_report_path'
            )
            self.model.training_class_evaluation_report_path = old_paths.get(
                'training_class_evaluation_report_path'
            )
            self.model.training_confusion_matrix_report_path = old_paths.get(
                'training_confusion_matrix_report_path'
            )
            self.model.training_confusion_matrix_plot_path = old_paths.get(
                'training_confusion_matrix_plot_path'
            )
            self.model.training_class_distribution_report_path = old_paths.get(
                'training_class_distribution_report_path'
            )
            self.model.training_class_distribution_plot_path = old_paths.get(
                'training_class_distribution_plot_path'
            )
            self.model.training_classified_point_cloud_path = old_paths.get(
                'training_classified_point_cloud_path'
            )
        # Restore random forest paths
        if isinstance(self.model, RandomForestClassificationModel):
            self.model.importance_report_path = old_paths.get(
                'importance_report_path'
            )
            self.model.decision_plot_path = old_paths.get(
                'decision_plot_path'
            )
        # Restore deep learning paths
        if self.model.is_deep_learning_model():
            self.restore_dl_model_paths(old_paths)

    def restore_dl_model_paths(self, old_paths):
        """
        Restore the model assuming it is a deep learning model. This method
        will not check that self.model is a deep learning model. Instead,
        it assumes it is because otherwise the method MSUT NOT be called.

        See :meth:`model_op.ModelOp.restore_model_paths`.

        :param old_paths: The output parameter. It must be a dictionary where
            any variable (key) that needs to restore its old path (value) is
            represented.
        :type old_paths: dict
        :return: Nothing at all, the old_paths dictionary is usted to restore
            the variables of the model to its previous value.
        """
        # Extract objects of interest
        model_args = self.model.model_args
        model_handling = model_args.get('model_handling', None)
        preproc = model_args.get('pre_processing', None)
        # Restore training activations path
        self.model.training_activations_path =\
            old_paths['training_activations_path']
        # Restore architecture graph path
        if model_args.get('architecture_graph_path', None) is not None:
            model_args['architecture_graph_path'] =\
                old_paths['architecture_graph_path']
        # Restore summary report path
        if model_handling.get('summary_report_path', None) is not None:
            model_handling['summary_report_path'] =\
                old_paths['summary_report_path']
        # Restore training history dir
        if model_handling.get('training_history_dir', None) is not None:
            model_handling['training_history_dir'] =\
                old_paths['training_history_dir']
        # Restore layer representation dirs
        if model_handling.get('features_structuring_representation_dir', None)\
                is not None:
            model_handling['features_structuring_representation_dir'] = \
                old_paths['feat_struct_repr_dir']
        if model_handling.get('rbf_feature_extraction_representation_dir', None)\
                is not None:
            model_handling['rbf_feature_extraction_representation_dir'] = \
                old_paths['rbf_feat_extract_repr_dir']
        if model_handling.get('rbf_feature_processing_representation_dir', None)\
                is not None:
            model_handling['rbf_feature_processing_representation_dir'] = \
                old_paths['rbf_feat_processing_repr_dir']
        if model_handling.get('kpconv_representation_dir', None) is not None:
            model_handling['kpconv_representation_dir'] = \
                old_paths['kpconv_representation_dir']
        if model_handling.get('skpconv_representation_dir', None) is not None:
            model_handling['skpconv_representation_dir'] = \
                old_paths['skpconv_representation_dir']
        # Restore checkpoint path
        if model_handling.get('checkpoint_path', None) is not None:
            model_handling['checkpoint_path'] = old_paths['checkpoint_path']
        # Restore training receptive fields distribution report path
        if preproc.get(
            'training_receptive_fields_distribution_report_path', None
        ) is not None:
            preproc['training_receptive_fields_distribution_report_path'] = \
                old_paths['training_receptive_fields_distribution_report_path']
        # Restore training receptive fields distribution plot path
        if preproc.get(
            'training_receptive_fields_distribution_plot_path', None
        ) is not None:
            preproc['training_receptive_fields_distribution_plot_path'] = \
                old_paths['training_receptive_fields_distribution_plot_path']
        # Restore training receptive fields dir
        if preproc.get('training_receptive_fields_dir', None) is not None:
            preproc['training_receptive_fields_dir'] = \
                old_paths['training_receptive_fields_dir']
        # Restore receptive fields distribution report path
        if preproc.get(
            'receptive_fields_distribution_report_path', None
        ) is not None:
            preproc['receptive_fields_distribution_report_path'] = \
                old_paths['receptive_fields_distribution_report_path']
        # Restore receptive fields distribution plot path
        if preproc.get(
            'receptive_fields_distribution_plot_path', None
        ) is not None:
            preproc['receptive_fields_distribution_plot_path'] = \
                old_paths['receptive_fields_distribution_plot_path']
        # Restore receptive fields dir
        if preproc.get('receptive_fields_dir', None) is not None:
            preproc['receptive_fields_dir'] = old_paths['receptive_fields_dir']
        # Restore training support points report path
        if preproc.get(
            'training_support_points_report_path', None
        ) is not None:
            preproc['training_support_points_report_path'] = \
                old_paths['training_support_points_report_path']
        if preproc.get('support_points_report_path', None) is not None:
            preproc['support_points_report_path'] = \
                old_paths['support_points_report_path']
        # Make the changes effective (on the arguments)
        if model_args is not None:
            self.model.model_args = model_args
        if model_handling is not None:
            self.model.model_args['model_handling'] = model_handling
        if preproc is not None:
            self.model.model_args['pre_processing'] = preproc
        # Make the changes effective on the path-related attributes
        if hasattr(self.model, "update_paths"):
            self.model.update_paths()

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
