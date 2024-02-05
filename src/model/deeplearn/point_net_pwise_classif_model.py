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
        PointNetPwiseClassifModel.update_pretrained_model(self, spec)

    @staticmethod
    def update_pretrained_model(dlmodel, spec):
        """
        See :meth:`PointNetPwiseClassifModel.overwrite_pretrained_model`.
        """
        # Overwrite training activations attributes
        spec_keys = spec.keys()
        if 'training_activations_path' in spec_keys:
            dlmodel.training_activations_path = spec['training_activations_path']
        # Overwrite model handler
        if 'model_args' in spec_keys:
            if not isinstance(dlmodel.model, DLModelHandler):
                raise DeepLearningException(
                    'Deep learning model cannot overwrite model handler '
                    'because it is not a DLModelHandler.'
                )
            dlmodel.model.overwrite_pretrained_model(spec['model_args'])

    def update_paths(self):
        """
        Consider the current specification of model args (self.model_args)
        to update the paths.
        """
        if self.model is not None:
            self.model.update_paths(self.model_args)

    def predict(self, pcloud, X=None, F=None, plots_and_reports=True):
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
        F = None
        if self.fnames is not None and len(self.fnames) > 0:
            F = pcloud.get_features_matrix(self.fnames)
        return self._predict(X, y=y, F=F, plots_and_reports=plots_and_reports)

    def get_input_from_pcloud(self, pcloud):
        """
        See :meth:`model.Model.get_input_from_pcloud`.
        """
        # TODO Rethink : Common impl. wrt RBFNetPwiseClassifModel.get_input_from_pcloud
        # No features
        if self.fnames is None:
            return pcloud.get_coordinates_matrix()
        # Features
        return [
            pcloud.get_coordinates_matrix(),
            pcloud.get_features_matrix(self.fnames)
        ]

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
        zhat, yhat = PointNetPwiseClassifModel.on_training_finished_predict(
            self, X, y, yhat
        )
        # Call parent's on_training_finished
        super().on_training_finished(X, y, yhat=yhat)
        # Evaluate computed predictions
        PointNetPwiseClassifModel.on_training_finished_evaluate(
            self, X, y, zhat, yhat
        )

    @staticmethod
    def on_training_finished_predict(dlmodel, X, y, yhat):
        """
        See :meth:`PointNetPwiseClassifModel.on_training_finished` and
        :meth:`PointNetPwiseClassifModel.on_training_finished_evaluate`.
        """
        start = time.perf_counter()
        zhat = None
        if yhat is None:
            zhat = []
            yhat = dlmodel._predict(X, F=None, y=y, zout=zhat)
            zhat = zhat[-1]
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'After-training deep learning model '
            f'computed in {end-start:.3f} seconds on training point cloud.'
        )
        return zhat, yhat

    @staticmethod
    def on_training_finished_evaluate(dlmodel, X, y, zhat, yhat):
        """
        See :meth:`PointNetPwiseClassifModel.on_training_finished` and
        :meth:`PointNetPwiseClassifModel.on_training_finished_predict`.
        """
        # Get the coordinates matrix even when [X, F] is given
        _X = X[0] if isinstance(X, list) else X
        # Write classified point cloud
        if dlmodel.training_classified_point_cloud_path is not None:
            ClassifiedPcloudReport(
                X=_X, y=y, yhat=yhat, zhat=zhat, class_names=dlmodel.class_names
            ).to_file(
                path=dlmodel.training_classified_point_cloud_path
            )
        # Write point-wise activations
        if dlmodel.training_activations_path is not None:
            start = time.perf_counter()
            activations = dlmodel.compute_pwise_activations(X)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'Point-wise activations of deep learning model computed in '
                f'{end-start:.3f} seconds.'
            )
            PwiseActivationsReport(
                X=_X, activations=activations, y=y
            ).to_file(
                path=dlmodel.training_activations_path
            )
            # ANOVA on activations
            start = time.perf_counter()
            Fval, pval = f_classif(activations, y)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'ANOVA computed on point-wise deep learning activations in '
                f'{end-start:.3f} seconds.'
            )
            BestScoreSelectionReport(
                fnames=None,
                scores=Fval,
                score_name='F-value',
                pvalues=pval,
                selected_features=None
            ).to_file(
                path=dlmodel.training_activations_path[
                     :dlmodel.training_activations_path.rfind('.')
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
        X = X if F is None else [X, F]
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
        # Compute and return
        return PointNetPwiseClassifModel.do_pwise_activations(
            self.model, remodel, X
        )

    @staticmethod
    def do_pwise_activations(model, remodel, X):
        """
        Assist the
        :meth:`PointNetPwiseClassifModel.compute_pwise_activations` method.
        """
        remodel.compile(
            **SimpleDLModelHandler.build_compilation_args(
                model.compilation_args
            )
        )
        # Compute the activations
        X_rf = model.arch.run_pre({'X': X})
        with tf.device("cpu:0"):
            start_cpu_activations = time.perf_counter()
            activations = remodel.predict(
                X_rf, batch_size=model.batch_size
            )
            end_cpu_activations = time.perf_counter()
            LOGGING.LOGGER.debug(
                "Activations computed on CPU in {t:.3f} seconds".format(
                    t=end_cpu_activations-start_cpu_activations
                )
            )
        # Propagate activations to original dimensionality
        rf = model.arch.pre_runnable.pre_processor \
            .last_call_receptive_fields
        propagated_activations = joblib.Parallel(
            n_jobs=model.arch.pre_runnable.pre_processor.nthreads
        )(
            joblib.delayed(
                rfi.propagate_values
            )(
                activations[i], reduce_strategy='mean'
            )
            for i, rfi in enumerate(rf)
        )
        # Reduce overlapping propagations to mean
        I = model.arch.pre_runnable.pre_processor \
            .last_call_neighborhoods
        npoints = X[0].shape[0] if isinstance(X, list) else X.shape[0]
        activations = GridSubsamplingPostProcessor.pwise_reduce(
            npoints, activations.shape[-1], I, propagated_activations
        )
        # Return
        return activations
