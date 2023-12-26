# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.arch.point_net_pwise_classif import \
    PointNetPwiseClassif
from src.model.deeplearn.handle.dl_model_handler import \
    DLModelHandler
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor
from src.report.classified_pcloud_report import ClassifiedPcloudReport
from src.report.pwise_activations_report import PwiseActivationsReport
from src.report.best_score_selection_report import BestScoreSelectionReport
from src.utils.dict_utils import DictUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
from sklearn.feature_selection import f_classif
import tensorflow as tf
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class PointNetPwiseClassifModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    PointNet model for point-wise classification tasks.
    See :class:`.ClassificationModel`.

    :ivar model: The deep learning model wrapped by the corresponding handler,
        i.e., the :class:`.PointNetPwiseClassif` model wrapped by a
        :class:`.SimpleDLModelHandler` handler.
    :vartype model: :class:`.DLModelHandler`
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        PointNetPwiseClassifModel from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            PointNetPwiseClassifModel.
        """
        # Initialize from parent
        kwargs = ClassificationModel.extract_model_args(spec)
        # Extract particular arguments for PointNetPwiseClassif models
        kwargs['training_activations_path'] = spec.get(
            'training_activations_path', None
        )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of PointNetPwiseClassifModel.

        :param kwargs: The attributes for the PointNetPwiseClassifModel that
            will also be passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the PointNetPwiseClassifModel
        self.model = None  # By default, internal model is not instantiated
        self.training_activations_path = kwargs.get(
            'training_activations_path', None
        )

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def is_deep_learning_model(self):
        """
        See :class:`.Model` and :meth:`model.Model.is_deep_learning_model`.
        """
        return True

    def prepare_model(self):
        """
        Prepare a PointNet point-wise classifier with current model arguments.

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        :rtype: :class:`.PointNetPwiseClassif`
        """
        # Instantiate model
        if self.model is None:
            if self.model_args is not None:
                self.model = PointNetPwiseClassif(**self.model_args)
            else:
                LOGGING.LOGGER.debug(
                    'Preparing a PointNetPwiseClassifModel with no model_args.'
                )
                self.model = PointNetPwiseClassif()
        else:
            LOGGING.LOGGER.debug(
                'Preparing a pretrained PointNetPwiseClassifModel.'
            )
            return self.model
        # Wrap model with handler
        self.model = SimpleDLModelHandler(
            self.model,
            compilation_args=self.model_args.get('compilation_args', None),
            class_names=self.class_names,
            **self.model_args.get('model_handling', None)
        )
        return self.model

    def overwrite_pretrained_model(self, spec):
        """
        See :meth:`model.Model.overwrite_pretrained_model`.
        """
        super().overwrite_pretrained_model(spec)
        # Overwrite training activations attributes
        spec_keys = spec.keys()
        if 'training_activations_path' in spec_keys:
            self.training_activations_path = spec['training_activations_path']
        # Overwrite model handler
        if 'model_args' in spec_keys:
            if not isinstance(self.model, DLModelHandler):
                raise DeepLearningException(
                    'PointNetPwiseClassifModel cannot overwrite model handler '
                    'because it is not a DLModelHandler.'
                )
            self.model.overwrite_pretrained_model(spec['model_args'])

    def update_paths(self):
        """
        Consider the current specification of model args (self.model_args)
        to update the paths.

        """
        if self.model is not None:
            # TODO Rethink : Delegate model_handling paths to model handler
            model_handling = self.model_args['model_handling']
            self.model.summary_report_path = \
                model_handling['summary_report_path']
            self.model.training_history_dir = \
                model_handling['training_history_dir']
            self.model.feat_struct_repr_dir = \
                model_handling['features_structuring_representation_dir']
            self.model.checkpoint_path = model_handling['checkpoint_path']
            if self.model.arch is not None:
                self.model.arch.architecture_graph_path = \
                    self.model_args['architecture_graph_path']
                # TODO Rethink : Delegate preproc paths to pre-processor
                pre_processor = None
                if self.model.arch.pre_runnable is not None:
                    if hasattr(self.model.arch.pre_runnable, "pre_processor"):
                        pre_processor = \
                            self.model.arch.pre_runnable.pre_processor
                if pre_processor is not None:
                    preproc = self.model_args['pre_processing']
                    pre_processor.training_receptive_fields_distribution_report_path = \
                        preproc['training_receptive_fields_distribution_report_path']
                    pre_processor.training_receptive_fields_distribution_plot_path = \
                        preproc['training_receptive_fields_distribution_plot_path']
                    pre_processor.training_receptive_fields_dir = \
                        preproc['training_receptive_fields_dir']
                    pre_processor.receptive_fields_distribution_report_path = \
                        preproc['receptive_fields_distribution_report_path']
                    pre_processor.receptive_fields_distribution_plot_path = \
                        preproc['receptive_fields_distribution_plot_path']
                    pre_processor.receptive_fields_dir = \
                        preproc['receptive_fields_dir']
                    pre_processor.training_support_points_report_path = \
                        preproc['training_support_points_report_path']
                    pre_processor.support_points_report_path = \
                        preproc['support_points_report_path']

    def predict(self, pcloud, X=None, F=None):
        """
        Use the model to compute predictions on the input point cloud.

        The behavior of the base implementation (see
        :meth:`model.Model.predict`) is extended to account for X as a
        coordinates matrix and to ignore F. In other words, this PointNet
        implementation does not support input features.

        :param X: The input matrix of coordinates where each row represents a
            point from the point cloud: If not given, it will be retrieved
            from the point cloud.
        :type X: :class:`np.ndarray`
        :param F: Ignored.
        """
        if X is None:
            X = pcloud.get_coordinates_matrix()
        y = None
        if pcloud.has_classes():
            y = pcloud.get_classes_vector()
        return self._predict(X, y=y, F=None)

    def get_input_from_pcloud(self, pcloud):
        """
        See :meth:`model.Model.get_input_from_pcloud`.
        """
        return pcloud.get_coordinates_matrix()

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y, F=None, info=True):
        """
        The fundamental training logic to train a PointNet-based point-wise
        classifier.

        See :class:`.ClassificationModel` and :class:`.Model`.
        Also see :meth:`model.Model.training`.

        :param F: Ignored.
        """
        # Initialize model instance
        self.prepare_model()
        # Train the model
        self.model = self.model.fit(X, y)

    def on_training_finished(self, X, y, yhat=None):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Compute predictions on training data
        start = time.perf_counter()
        zhat = None
        if yhat is None:
            zhat = []
            yhat = self._predict(X, F=None, y=y, zout=zhat)
            zhat = zhat[-1]
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'PointNet point-wise classification on training point cloud '
            f'computed in {end-start:.3f} seconds.'
        )
        # Training evaluation
        super().on_training_finished(X, y, yhat=yhat)
        # Write classified point cloud
        if self.training_classified_point_cloud_path is not None:
            ClassifiedPcloudReport(
                X=X, y=y, yhat=yhat, zhat=zhat, class_names=self.class_names
            ).to_file(
                path=self.training_classified_point_cloud_path
            )
        # Write point-wise activations
        if self.training_activations_path is not None:
            start = time.perf_counter()
            activations = self.compute_pwise_activations(X)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'PointNet point-wise activations computed in '
                f'{end-start:.3f} seconds.'
            )
            PwiseActivationsReport(
                X=X, activations=activations, y=y
            ).to_file(
                path=self.training_activations_path
            )
            # ANOVA on activations
            start = time.perf_counter()
            Fval, pval = f_classif(activations, y)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'ANOVA computed on PointNet point-wise activations in '
                f'{end-start:.3f} seconds.'
            )
            BestScoreSelectionReport(
                fnames=None,
                scores=Fval,
                score_name='F-value',
                pvalues=pval,
                selected_features=None
            ).to_file(
                path=self.training_activations_path[
                    :self.training_activations_path.rfind('.')
                ] + '_ANOVA.csv'
            )

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, F=None, y=None, zout=None, plots_and_reports=True):
        """
        Extend the base _predict method.

        See :meth:`model.Model_predict`.

        :param y: The expected point-wise labels/classes. It can be used by
            predictions on training data to generate a thorough representation
            of the receptive fields.
        """
        return self.model.predict(
            X, y=y, zout=zout, plots_and_reports=plots_and_reports
        )

    # ---  POINT NET PWISE CLASSIF METHODS  --- #
    # ----------------------------------------- #
    def compute_pwise_activations(self, X):
        """
        Compute the point wise activations of the last layer before the
        output softmax layer in the PointNet-based point-wise classification
        model.

        :param X: The matrix of coordinates representing the point cloud.
        :type X: :class:`np.ndarray`
        :return: The matrix of point wise activations where points are rows
            and the columns are the components of the output activation
            function (activated vector or point-wise features).
        :rtype: :class:`np.ndarray`
        """
        # Prepare model to compute activations
        remodel = tf.keras.Model(
            inputs=self.model.compiled.inputs,
            outputs=self.model.compiled.get_layer(index=-2).output
        )
        remodel.compile(
            **SimpleDLModelHandler.build_compilation_args(
                self.model.compilation_args
            )
        )
        # Compute the activations
        X_rf = self.model.arch.run_pre({'X': X})
        with tf.device("cpu:0"):
            start_cpu_activations = time.perf_counter()
            activations = remodel.predict(
                X_rf, batch_size=self.model.batch_size
            )
            end_cpu_activations = time.perf_counter()
            LOGGING.LOGGER.debug(
                "Activations computed on CPU in {t:.3f} seconds".format(
                    t=end_cpu_activations-start_cpu_activations
                )
            )
        # Propagate activations to original dimensionality
        rf = self.model.arch.pre_runnable.pre_processor\
            .last_call_receptive_fields
        propagated_activations = joblib.Parallel(
            n_jobs=self.model.arch.pre_runnable.pre_processor.nthreads
        )(
            joblib.delayed(
                rfi.propagate_values
            )(
                activations[i], reduce_strategy='mean'
            )
            for i, rfi in enumerate(rf)
        )
        # Reduce overlapping propagations to mean
        I = self.model.arch.pre_runnable.pre_processor\
            .last_call_neighborhoods
        activations = GridSubsamplingPostProcessor.pwise_reduce(
            X.shape[0], activations.shape[-1], I, propagated_activations
        )
        # Return
        return activations
