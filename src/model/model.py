# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
from src.utils.imputer_utils import ImputerUtils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ModelException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to model components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Model:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface governing any model.

    :ivar training_type: The type of training. Either "base", "autoval",
        or "stratified_kfold".
    :vartype training_type: str
    :ivar autoval_size: The size of the auto validation set (used by
        :meth:`model.Model.train_autoval`). It can be given as a ratio in
        [0, 1] or as the set cardinality in [0, m].
    :vartype autoval_size: int or float
    :ivar shuffle_points: Flag governing whether to shuffle the points (True)
        or not (False). It is used by :meth:`model.Model.train_autoval` and
        :meth:`model.Model.train_stratified_kfold`.
    :vartype shuffle_points: bool
    :ivar num_folds: The number of folds, i.e. K for K-Folding.
    :vartype num_folds: int
    :ivar imputer: The imputer to deal with missing values (can be None).
    :vartype imputer: :class:`.Imputer`
    :ivar fnames: The list of feature names (fnames) attribute. These features
        must correspond to the features in the input point cloud for
        training and predictions.
    :vartype fnames: list
    :ivar random_seed: Optional attribute to specify a fixed random seed for
        the random computations of the model.
    :vartype random_seed: int
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a Model from a
        key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a Model.
        """
        # Initialize
        kwargs = {
            'training_type': spec.get('training_type', None),
            'random_seed': spec.get('random_seed', None),
            'shuffle_points': spec.get('shuffle_points', None),
            'autoval_size': spec.get('autoval_size', None),
            'num_folds': spec.get('num_folds', None),
            'imputer': spec.get('imputer', None),
            'fnames': spec.get('fnames', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Root initialization for any instance of type Model.

        :param kwargs: The attributes for the Model.
        """
        # Fundamental initialization of any model
        self.training_type = kwargs.get("training_type", "base")
        self.random_seed = kwargs.get("random_seed", None)
        self.shuffle_points = kwargs.get("shuffle_points", True)
        self.autoval_size = kwargs.get("autoval_size", 0.2)
        self.num_folds = kwargs.get("num_folds", 5)
        self.imputer = kwargs.get("imputer", None)
        if self.imputer is not None:
            imputer_class = ImputerUtils.extract_imputer_class(self.imputer)
            self.imputer = imputer_class(
                **imputer_class.extract_imputer_args(self.imputer)
            )
        self.fnames = kwargs.get("fnames", None)
        if self.fnames is None:
            raise ModelException(
                "No feature names were specified for the model."
            )

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def train(self, pcloud):
        """
        Train the model from a given input point cloud.

        :param pcloud: The input point cloud to train the model with.
        :return: The trained model.
        :rtype: :class:`.Model`
        """
        trainT = self.training_type.lower()
        if trainT == 'base':
            return self.train_base(pcloud)
        elif trainT == 'autoval':
            return self.train_autoval(pcloud)
        elif trainT == 'stratified_kfold':
            return self.train_stratified_kfold(pcloud)
        raise ModelException(f'Unexpected training type: {self.training_type}')

    def predict(self, pcloud):
        """
        Use the model to compute predictions on the input point cloud.

        :param pcloud: The input point cloud to compute the predictions.
        :return: The point-wise predictions.
        :rtype: :class:`np.ndarray`
        """
        X = pcloud.get_features_matrix(self.fnames)  # Often named F instead
        if self.imputer is not None:
            X = self.imputer.impute(X)
        return self._predict(X)

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    @abstractmethod
    def training(self, X, y):
        """
        The fundamental training logic defining the model.

        It must be overridden by non-abstract subclasses.

        :param X: The input matrix representing the point cloud, e.g., the
            geometric features matrix.
        :param y: The class for each point.
        :return: Nothing, but the model itself is updated.
        """
        pass

    def autoval(self, y, yhat):
        """
        Auto validation during training.

        Any non-abstract subclass must provide an implementation for autoval
        or avoid calling :meth:`model.Model.train_autoval` and
        `model.Model.train_stratified_kfold` to prevent exceptions/errors.

        :param y: The expected values.
        :param yhat: The predicted values.
        :return: The results of the auto validation.
        """
        raise ModelException(
            'Auto validation is not supported by the current model.'
        )

    def train_base(self, pcloud):
        """
        Straightforward model training.

        :param pcloud: The input point cloud to train the model.
        :return: The trained model.
        :rtype: :class:`.Model`
        """
        X = pcloud.get_features_matrix(self.fnames)
        y = pcloud.get_classes_vector()
        if self.imputer is not None:
            X, y = self.imputer.impute(X, y)
        self.training(X, y)
        return self

    def train_autoval(self, pcloud):
        """
        Auto validation training strategy.

        Some part of the data is ignored during training to be used for
        later validation of the trained model (auto validation).

        The subsets are extracted in a stratified way. See
        :meth:`model.Model.train_stratified_kfold` for a description of
        stratification.

        :param pcloud: The input point cloud to train the model.
        :return: The trained model.
        :rtype: :class:`.Model`
        """
        X = pcloud.get_features_matrix(self.fnames)
        y = pcloud.get_classes_vector()
        if self.imputer is not None:
            X, y = self.imputer.impute(X, y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y,
            test_size=self.autoval_size,
            shuffle=self.shuffle_points,
            stratify=y,
            random_state=self.random_seed
        )
        self.training(Xtrain, ytrain)
        # TODO Rethink : Auto validation
        return self

    def train_stratified_kfold(self, pcloud):
        """
        Stratified k-fold training strategy.

        Stratification consists of dividing the data into subsets (strata).
        Test points are taken from each subset, i.e., each stratum of the
        strata, so they follow a class distribution approximately proportional
        to the original set. This strategy guarantees that the test points
        constitute a reliable representation of the original points.

        K-folding consists of dividing the data into K different subsets
        (folds). Then, K iterations are computed such that at each one a
        different fold is considered as the test set and the other (K-1) folds
        are used for training.

        Stratified K-folding is K-folding with stratified folds, i.e., each
        fold is also a stratum.

        :param pcloud: The input point cloud to train the model.
        :return: The trained model.
        :rtype: :class:`.Model`
        """
        X = pcloud.get_features_matrix(self.fnames)
        y = pcloud.get_classes_vector()
        if self.imputer is not None:
            X, y = self.imputer.impute(X, y)
        self.training(X, y)
        skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=self.shuffle_points,
            random_state=self.random_seed
        )
        for i, (Itrain, Ival) in enumerate(skf.split(X, y)):
            Xi, yi = X[Itrain], y[Ival]
            self.training(Xi, yi)
            # TODO Rethink : Auto validation (append to list for variability)
        return self

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    @abstractmethod
    def _predict(self, X):
        """
        Predict the points represented by the matrix X. Typically, it will be
        a matrix of features.

        :param X: The input matrix representing the point cloud, e.g., the
            geometric features matrix.
        :return: The point-wise predictions.
        :rtype: :class:`np.ndarray`
        """
        pass
