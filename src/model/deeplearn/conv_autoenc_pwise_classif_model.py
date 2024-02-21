# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.arch.conv_autoenc_pwise_classif import \
    ConvAutoencPwiseClassif
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ConvAutoencPwiseClassifModel(ClassificationModel):
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
        Initialize an instance of ConvAutoencPwiseClassifModel.

        :param kwargs: The attributes for the ConvAutoencPwiseClassifModel
            that will also be passed to the parent.
        """
        # Call parent init
        if not 'fnames' in kwargs:
            if 'model_args' in kwargs and 'fnames' in kwargs['model_args']:
                kwargs['fnames'] = kwargs['model_args']['fnames']
            else:
                kwargs['fnames'] = []  # Avoid None fnames exception for ConvAutoenc
        super().__init__(**kwargs)
        # Basic attributes of the ConvAutoencPwiseClassifModel
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
                    "Preparing a ConvAutoencPwiseClassifModel with no "
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
        # PointNet, RBFNet and ConvAutoenc.
        # Abstract it to a common logic (DRY, implement only once)
        super().overwrite_pretrained_model(spec)
        PointNetPwiseClassifModel.update_pretrained_model(self, spec)

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
        P = self.get_input_from_pcloud(pcloud)
        if X is not None:
            P[0] = pcloud.get_coordinates_matrix()
        if F is not None:
            if len(P) > 1:
                P[1] = F
            else:
                P.append(F)
        return self._predict(P[0], F=P[1])

    def get_input_from_pcloud(self, pcloud):
        """
        See :meth:`model.Model.get_input_from_pcloud`.
        """
        X = pcloud.get_coordinates_matrix()
        # Return without features
        if self.fnames is None:
            return [
                X,
                np.ones((X.shape[0], 1))  # ConvAutoenc always needs features
            ]
        # Handle ones as features
        if 'ones' in self.fnames:
            self.fnames.remove('ones')
            F = np.hstack([
                np.ones((X.shape[0], 1)),
                pcloud.get_features_matrix(self.fnames)
            ]) if len(self.fnames) > 0 else np.ones((X.shape[0], 1))
            self.fnames.append('ones')
        else:  # Handle features without ones
            F = pcloud.get_features_matrix(self.fnames)
        # Return with features
        return [X, F]

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
        self.model = self.model.fit(X, y)
        end = time.perf_counter()
        # Log end of execution
        if info:
            LOGGING.LOGGER.info(
                'ConvAutoencPwiseClassifModel trained in '
                f'{end-start:.3f} seconds.'
            )

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
