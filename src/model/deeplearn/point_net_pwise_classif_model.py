# ---   IMPORTS   --- #
# ------------------- #
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.arch.point_net_pwise_classif import \
    PointNetPwiseClassif
from src.model.deeplearn.handle.simple_dl_model_handler import \
    SimpleDLModelHandler
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class PointNetPwiseClassifModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    PointNet model for point-wise classification tasks.
    See :class:`.ClassificationModel`.

    # TODO Rethink : Doc internal variables (member attributes)
    """
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
        # TODO Rethink : Implement

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def prepare_model(self):
        """
        Prepare a PointNet point-wise classifier with current model arguments.

        :return: The prepared model itself. Note it is also assigned as the
            model attribute of the object/instance.
        :rtype: :class:`.PointNetPwiseClassif`
        """
        # Instantiate model
        if self.model_args is not None:
            self.model = PointNetPwiseClassif(**self.model_args)
        else:
            LOGGING.LOGGER.debug(
                'Preparing a PointNetPwiseClassifModel with no model_args'
            )
            self.model = PointNetPwiseClassif()
        # Wrap model with handler
        self.model = SimpleDLModelHandler(
            self.model,
            compilation_args=self.model_args.get('compilation_args', None),
            **self.model_args.get('model_handling', None)
        )
        return self.model

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
        return self._predict(X, F=None)

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

        :param F: Ignored
        """
        # Initialize model instance
        self.prepare_model()
        # Train the model
        start = time.perf_counter()
        self.model = self.model.fit(X, y)
        end = time.perf_counter()
        # TODO Rethink : Implement
        pass

    def on_training_finished(self, X, y):
        """
        See :meth:`model.Model.on_training_finished`.
        """
        # Report scores for trained model
        # TODO Rethink : Implement
        # Report point-wise activation maps
        # TODO Rethink : Implement
        # Plot point-wise activation
        # TODO Rethink : Implement
        pass

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, F=None):
        """
        Extend the base _predict method to account for coordinates (X) as
        input.

        See :meth:`model.Model_predict`.
        """
        return self.model.predict(X)
