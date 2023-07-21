# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class ConvAutoencClassificationModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    Convolutional autoencoder model for classification tasks.
    See :class:`.ClassificationModel`.

    # TODO Rethink : Doc internal variables (member attributes)
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ConvAutoencClassificationModel.

        :param kwargs: The attributes for the ConvAutoencClassificationModel
            that will also be passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the ConvAutoencClassificationModel
        # TODO Rethink : Implement

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def prepare_model(self):
        """
        Prepare a convolutional autoencoder point-wise classifier with current
        model arguments.

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        :rtype: :class:`.ConvAutoencPwiseClassif
        """
        # Instantiate the model
        if self.model_args is not None:
            self.model = ConvAutoencPwiseClassif(**self.model_args)
        else:
            LOGGING.LOGGER.info(
                "Preparing a ConvAutoencClassificationModel with no "
                "`model_args`"
            )
            self.model = ConvAutoencPwiseClassif()
        # Wrap model with handler
        self.model = SimpleDLModelHandler(self.model)
        return self.model

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
        self.model = self.model.fit(X, y, F=F)
        end = time.perf_counter()
        # Log end of execution
        if info:
            LOGGING.LOGGER.info(
                'ConvAutoencClassificationModel trained in '
                f'{end-start:.3f} seconds.'
            )

    def on_training_finished(self, X, y):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Retrieve F from object's cache
        F = self.cache_F
        self.cache_F = None
        # Report scores for trained model
        # TODO Rethink : Implement
        # Report point-wise activation maps
        # TODO Rethink : Implement
        # Plot point-wise activation
        # TODO Rethink : Implement

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, F=None):
        """
        Extend the base _predict method to account for coordinates (X) and
        features (F) separately.

        See :meth:`model.Model._predict`.
        """
        return self.model.predict(X, F=F)
