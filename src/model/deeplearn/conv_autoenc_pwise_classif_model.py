# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.arch.conv_autoenc_pwise_classif import \
    ConvAutoencPwiseClassif
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
from src.model.deeplearn.dlrun.grid_subsampling_post_processor import \
    GridSubsamplingPostProcessor
from src.report.classified_pcloud_report import ClassifiedPcloudReport
from src.report.pwise_activations_report import PwiseActivationsReport
from src.report.best_score_selection_report import BestScoreSelectionReport
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from sklearn.feature_selection import f_classif
import joblib
import tensorflow as tf
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ConvAutoencClassificationModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    Convolutional autoencoder model for classification tasks.
    See :class:`.ClassificationModel`.

    :ivar model: The deep learning model wrapped by the corresponding handler,
        i.e., the :class:`.ConvAutoencPwiseClassif` model wrapped by a
        :class:`.SimpleDLModelHandler` handler.
    :vartype model: :class:`.DLModelHandler`
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        ConvAutoencPwiseClassifModel from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            ConvAutoencPwiseClassifModel.
        """
        # Initialize from parent
        kwargs = ClassificationModel.extract_model_args(spec)
        # Extract particular arguments for ConvAutoencPwiseClassif models
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
        Initialize an instance of ConvAutoencClassificationModel.

        :param kwargs: The attributes for the ConvAutoencClassificationModel
            that will also be passed to the parent.
        """
        # Call parent init
        if not 'fnames' in kwargs:
            if 'model_args' in kwargs and 'fnames' in kwargs['model_args']:
                kwargs['fnames'] = kwargs['model_args']['fnames']
            else:
                kwargs['fnames'] = []  # Avoid None fnames exception for ConvAutoenc
        super().__init__(**kwargs)
        # Basic attributes of the ConvAutoencClassificationModel
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
        Prepare a convolutional autoencoder point-wise classifier with current
        model arguments.

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        :rtype: :class:`.ConvAutoencPwiseClassif`
        """
        # Instantiate the model
        if self.model is None:
            if self.model_args is not None:
                self.model = ConvAutoencPwiseClassif(**self.model_args)
            else:
                LOGGING.LOGGER.info(
                    "Preparing a ConvAutoencClassificationModel with no "
                    "`model_args`"
                )
                self.model = ConvAutoencPwiseClassif()
        else:
            LOGGING.LOGGER.debug(
                'Preparing a pretrained ConvAutoencPwiseClassifModel.'
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
        # TODO Rethink : In general training_activations_path is used for
        # PointNet, RBFNet and ConvAutoenc.
        # Abstract it to a common logic (DRY, implement only once)
        super().overwrite_pretrained_model(spec)
        # Overwrite training activations attributes
        spec_keys = spec.keys()
        if 'training_activations_path' in spec_keys:
            self.training_activations_path = spec['training_activations_path']
        # Overwrite model handler
        if 'model_args' in spec_keys:
            if not isinstance(self.model, DLModelHandler):
                raise DeepLearningException(
                    'RBFNetPwiseClassifModel cannot overwrite model handler '
                    'because it is not a DLModelHandler.'
                )
            self.model.overwrite_pretrained_model(spec['model_args'])

    def update_paths(self):
        """
        Consider the current specification of model args (self.model_args)
        to update the paths.
        """
        if self.model is not None:
            self.model.update_paths(self.model_args)

    def predict(self, pcloud, X=None, F=None):
        """
        Use the model to compute predictions on the input point cloud.

        The behavior of the base implementation (see
        :meth:`model.Model.predict`) is extended to account for X and F
        matrix as different entities.

        :param X: The input matrix of coordinates where each row represents a
            point from the point cloud (OPTIONAL). If not given, it will be
            retrieved from the point cloud.
        :type X: :class:`np.ndarray`
        :param F: The input matrix of features (OPTIONAL). If not given, it
            will be retrieved from the point cloud if there are feature names
            (fnames= available.
        :type F: :class:`np.ndarray`
        """
        if X is None:
            X = pcloud.get_coordinates_matrix()
        if F is None and self.fnames is not None:
            F = pcloud.get_features_matrix(self.fnames)
        return self._predict(X, F=F)

    def get_input_from_pcloud(self, pcloud):
        """
        See :meth:`model.Model.get_input_from_pcloud`.
        """
        # No features
        if self.fnames is None:
            X = pcloud.get_coordinates_matrix()
            return [
                X,
                np.ones((X.shape[0], 1))  # ConvAutoenc always needs features
            ]
        # Features
        return [
            pcloud.get_coordinates_matrix(),
            pcloud.get_features_matrix(self.fnames)
        ]

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y, F=None, info=True):
        """
        The fundamental training logic to train a convolutional autoencoder
        point-wise classifier.

        See :class:`.ClassificationModel` and :class:`.Model`.
        Also see :meth:`model.Model.training`.

        :param F: An optional (can be None) matrix of input features.
        :type F: :class:`np.ndarray`
        """
        # Initialize model instance
        self.prepare_model()
        # Train the model
        start = time.perf_counter()
        self.cache_F = F  # Cache F for on training finished
        self.model = self.model.fit(X, y)
        end = time.perf_counter()
        # Log end of execution
        if info:
            LOGGING.LOGGER.info(
                'ConvAutoencClassificationModel trained in '
                f'{end-start:.3f} seconds.'
            )

    def on_training_finished(self, X, y, yhat=None):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Retrieve F from object's cache
        # TODO Rethink : Is cache_F necessary? remove it?
        F = self.cache_F
        self.cache_F = None
        # TODO Rethink : Logic below (not above) to common impl. wrt PNet and RBFNet ?
        # Compute predictions on training data
        start = time.perf_counter()
        zhat = None
        if yhat is None:
            zhat = []
            yhat = self._predict(X, F=None, y=y, zout=zhat)
            zhat = zhat[-1]
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'ConvAutoenc point-wise classification on training point cloud '
            f'computed in {end-start:.3f} seconds.'
        )
        # Training evaluation
        super().on_training_finished(X, y, yhat=yhat)
        # Get the coordinates matrix even when [X, F] is given
        _X = X[0] if isinstance(X, list) else X
        # Write classified point cloud
        if self.training_classified_point_cloud_path is not None:
            ClassifiedPcloudReport(
                X=_X, y=y, yhat=yhat, zhat=zhat, class_names=self.class_names
            ).to_file(
                path=self.training_classified_point_cloud_path
            )
        # Write point-wise activations
        if self.training_activations_path is not None:
            start = time.perf_counter()
            activations = self.compute_pwise_activations(X)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'ConvAutoenc point-wise activations computed in '
                f'{end-start:.3f} seconds.'
            )
            PwiseActivationsReport(
                X=_X, activations=activations, y=y
            ).to_file(
                path=self.training_activations_path
            )
            # ANOVA on activations
            start = time.perf_counter()
            Fval, pval = f_classif(activations, y)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'ANOVA computed on ConvAutoenc point-wise activations in '
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

        See :meth:`model.Model._predict`.

        :param y: The expected point-wise labels/classes. It can be used by
            predictions on training data to generate a thorough representation
            of the receptive fields.
        """
        if F is None:
            if len(X) < 2:
                raise DeepLearningException(
                    'ConvAutoencPwiseClassifModel cannot work without features. '
                    'At least a column-wise vector of features must be given.'
                )
            else:
                P = X
        else:
            P = [X, F]
        return self.model.predict(
            P, y=y, zout=zout, plots_and_reports=plots_and_reports
        )

    # ---  RBFNET PWISE CLASSIF METHODS  --- #
    # -------------------------------------- #
    def compute_pwise_activations(self, X):
        """
        Compute the point-wise activations of the last layer before the output
        softmax (or sigomid for binary classification) layer in the
        convolutional autoencoder point-wise classification model.

        :param X: The matrix of coordinates representing the point cloud.
            Alternatively, it can be a list such that X[0] is the matrix of
            coordinates and X[1] the matrix of features.
        :type X: :class:`np.ndarray` or list
        :return: The matrix of point-wise activations where points are rows
            and the columns are the components of the output activation
            function (activated vector or point-wise features).
        :rtype: :class:`np.ndarray`
        """
        # Prepare model to compute activations
        remodel = tf.keras.Model(
            inputs=self.model.compiled.inputs,
            outputs=self.model.compiled.get_layer(index=-2).output
        )
        return PointNetPwiseClassifModel.do_pwise_activations(
            self.model, remodel, X
        )
