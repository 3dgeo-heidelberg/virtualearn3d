from src.model.classification_model import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import time


# ---   CLASS   --- #
# ----------------- #
class RandomForestClassificationModel(ClassificationModel):
    """
    :author: Alberto M. Esmoris Pena

    RandomForest model.
    See :class:`.Model`

    :ivar model_args: The arguments to initialize a new RandomForest model.
    :vartype model_args: dict
    :ivar model: The internal representation of the model.
    :vartype model: :class:`RandomForestClassifier`
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
        kwargs['model_args'] = spec.get('model_args', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of RandomForestModel.

        :param kwargs:  The attributes for the RandomForestClassificationModel
            that will also be passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the RandomForestClassificationModel
        self.model_args = kwargs.get("model_args", None)
        self.model = None

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y):
        """
        The fundamental training logic to train a random forest classifier.

        :param X: The input matrix representing the point cloud, e.g., the
            geometric features matrix.
        :param y: The class for each point.
        :return: Nothing, but the model itself is updated.
        """
        # Initialize model instance
        if self.model_args is not None:
            self.model = RandomForestClassifier(**self.model_args)
        else:
            LOGGING.LOGGER.info(
                "Training RandomForestClassificationModel with no `model_args`"
            )
            self.model = RandomForestClassifier()
        # Train the model
        start = time.perf_counter()
        self.model = self.model.fit(X, y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'RandomForestClassificationModel trained in'
            f'{end-start:.3f} seconds'
        )

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X):
        """
        See :meth:`model.Model._predict`
        """
        return self.model.predict(X)
