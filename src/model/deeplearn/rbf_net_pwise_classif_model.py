# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.arch.rbfnet_pwise_classif import \
    RBFNetPwiseClassif
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
from src.report.classified_pcloud_report import ClassifiedPcloudReport
from src.report.pwise_activations_report import PwiseActivationsReport
from src.report.best_score_selection_report import BestScoreSelectionReport
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from sklearn.feature_selection import f_classif
import tensorflow as tf
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class RBFNetPwiseClassifModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    RBFNet model for point-wise classification tasks.
    See :class:`.ClassificationModel`.

    :ivar model: The deep learning model wrapped by the corresponding handler,
        i.e., the :class:`.RBFNetPwiseClassif` model wrapped by a
        :class:`.SimpleDLModelHandler` handler.
    :vartype model: :class:`.DLModelHandler`
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        RBFNetPwiseClassifModel from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            RBFNetPwiseClassifModel.
        """
        # Initialize from parent
        kwargs = ClassificationModel.extract_model_args(spec)
        # Extract particular arguments for RBFNetPwiseClassif models
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
        Initialize an instance of RBFNetPwiseClassifModel.

        :param kwargs: The attributes for the RBFNetPwiseClassifModel that
            will also be passed to the parent.
        """
        # Call parent init
        if not 'fnames' in kwargs:
            if 'model_args' in kwargs and 'fnames' in kwargs['model_args']:
                kwargs['fnames'] = kwargs['model_args']['fnames']
            else:
                kwargs['fnames'] = []  # Avoid None fnames exception for RBFNet
        super().__init__(**kwargs)
        # Basic attributes of the RBFNetPwiseClassifModel
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
        Prepare a RBFNet point-wise classifier with current model arguments.

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        :rtype: :class:`.RBFNetPwiseClassif`
        """
        # Instantiate model
        if self.model is None:
            if self.model_args is not None:
                self.model = RBFNetPwiseClassif(**self.model_args)
            else:
                LOGGING.LOGGER.debug(
                    'Preparing a RBFNetPwiseClassifModel with no model_args.'
                )
                self.model = RBFNetPwiseClassif()
        else:
            LOGGING.LOGGER.debug(
                'Preparing a pretrained RBFNetPwiseClassifModel.'
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
        coordinates matrix and to ignore F. In other words, this RBFNet
        implementation does not support input features.

        :param pcloud: The input point cloud
        :type pcloud: :class:`.PointCloud`
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
        The fundamental training logic to train a RBFNet-based point-wise
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

    # ---  RBFNET PWISE CLASSIF METHODS  --- #
    # -------------------------------------- #
    def compute_pwise_activations(self, X):
        """
        Compute the point wise activations of the last layer before the
        output softmax layer in the RBFNet-based point-wise classification
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
