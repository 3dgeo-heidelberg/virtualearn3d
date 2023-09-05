# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
from src.utils.imputer_utils import ImputerUtils
from src.utils.tuner_utils import TunerUtils
from src.eval.kfold_evaluator import KFoldEvaluator
import src.main.main_logger as LOGGING
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ModelException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to model components.
    See :class:`.VL3DException`.
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
    :ivar stratkfold_report_path: The path where the report representing the
        evaluation of the k-folding procedure must be written.
    :vartype stratkfold_report_path: str
    :ivar stratkfold_plot_path: The path where the plot representing the
        evaluation of the k-folding procedure must be written.
    :vartype stratkfold_plot_path: str
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
            'hyperparameter_tuning': spec.get('hyperparameter_tuning', None),
            'fnames': spec.get('fnames', None),
            'stratkfold_report_path': spec.get('stratkfold_report_path', None),
            'stratkfold_plot_path': spec.get('stratkfold_plot_path', None),
            'model_args': spec.get('model_args', None)
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
        self.autoval_metrics_names = kwargs.get('autoval_metrics', None)
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
        self.hypertuner = kwargs.get("hyperparameter_tuning", None)
        if self.hypertuner is not None:
            hypertuner_class = TunerUtils.extract_tuner_class(self.hypertuner)
            self.hypertuner = hypertuner_class(
                **hypertuner_class.extract_tuner_args(self.hypertuner)
            )
        # Get feature names straight forward
        self.fnames = kwargs.get("fnames", None)
        # Validate feature names
        if self.fnames is None:
            raise ModelException(
                "No feature names were specified for the model."
            )
        self.stratkfold_report_path = kwargs.get(
            'stratkfold_report_path', None
        )
        self.stratkfold_plot_path = kwargs.get('stratkfold_plot_path', None)
        self.model_args = kwargs.get("model_args", None)

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def train(self, pcloud):
        """
        Train the model from a given input point cloud.

        :param pcloud: The input point cloud to train the model with.
        :return: The trained model.
        :rtype: :class:`.Model`
        """
        # Choose training method
        trainT = self.training_type.lower()
        if trainT == 'base':
            training_method = self.train_base
        elif trainT == 'autoval':
            training_method = self.train_autoval
        elif trainT == 'stratified_kfold':
            training_method = self.train_stratified_kfold
        else:
            raise ModelException(
                f'Unexpected training type: {self.training_type}'
            )
        # Tune hyperparameters (model_args might be replaced)
        if self.hypertuner is not None:
            self.prepare_model()
            self.hypertuner.tune(self, pcloud)
        # The training itself
        return training_method(pcloud)

    def predict(self, pcloud, X=None):
        """
        Use the model to compute predictions on the input point cloud.

        :param pcloud: The input point cloud to compute the predictions.
        :param X: The input matrix of features where each row represents a
            point from the point cloud (OPTIONAL). If given, X will be used as
            point-wise features instead of pcloud. It is often named F in the
            context of point clouds where the point cloud is a block matrix
            P = [X|F].
        :return: The point-wise predictions.
        :rtype: :class:`np.ndarray`
        """
        if X is None:
            X = pcloud.get_features_matrix(self.fnames)  # Often named F instead
        if self.imputer is not None:
            X = self.imputer.impute(X)
        return self._predict(X)

    @abstractmethod
    def prepare_model(self):
        """
        Prepare the model so its model attribute (i.e., model.model) can be
        used, for instance by a hyperparameter tuner.

        :return: The prepared model itself. However, the prepared model must be
            automatically assigned as an attribute of the object/instance too.
        """
        pass

    def overwrite_pretrained_model(self, spec):
        """
        This method must be called when preparing a pretrained model to
        overwrite any attribute that must be overriding depending on the model
        and the given training specification.

        :param spec: The key-word training specification to continue the
            training of the pretrained model.
        :type spec: dict
        :return: Nothing, but the model object internal state is updated.
        """
        spec_keys = spec.keys()
        # Overwrite autoval attributes
        if 'autoval_metrics' in spec_keys:
            self.autoval_metrics_names = spec['autoval_metrics']
        # Overwrite stratified kfolding attributes
        if 'stratkfold_report_path' in spec_keys:
            self.stratkfold_report_path = spec['stratkfold_report_path']
        if 'stratkfold_plot_path' in spec_keys:
            self.stratkfold_plot_path = spec['stratkfold_plot_path']
        # Overwrite hyperparameter tuning attributes
        if 'hyperparameter_tuning' in spec_keys:
            ht_spec = spec['hyperparameter_tuning']
            hypertuner_class = TunerUtils.extract_tuner_class(ht_spec)
            self.hypertuner = hypertuner_class(
                **hypertuner_class.extract_tuner_args(ht_spec)
            )

    def get_input_from_pcloud(self, pcloud):
        """
        Obtain the model-ready input from the given point cloud.

        :param pcloud: The point cloud containing the data to fit the model.
        :return: Model-ready input data.
        """
        return pcloud.get_features_matrix(self.fnames)

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    @abstractmethod
    def training(self, X, y, info=True):
        """
        The fundamental training logic defining the model.

        It must be overridden by non-abstract subclasses.

        :param X: The input matrix representing the point cloud, e.g., the
            geometric features matrix.
        :param y: The class for each point.
        :param info: True to enable info log messages, False otherwise.
        :return: Nothing, but the model itself is updated.
        """
        pass

    def autoval(self, y, yhat, info=True):
        """
        Auto validation during training.

        Any non-abstract subclass must provide an implementation for autoval
        or avoid calling :meth:`model.Model.train_autoval` and
        `model.Model.train_stratified_kfold` to prevent exceptions/errors.

        :param y: The expected values.
        :param yhat: The predicted values.
        :param info: True to log an info message with the auto validation,
            False otherwise.
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
        X = self.get_input_from_pcloud(pcloud)
        y = pcloud.get_classes_vector()
        if self.imputer is not None:
            X, y = self.imputer.impute(X, y)
        self.training(X, y)
        self.on_training_finished(X, y)
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
        # Training
        X = self.get_input_from_pcloud(pcloud)
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
        # Auto validation
        yhat_test = self.predict(None, X=Xtest)
        self.autoval(ytest, yhat_test)
        # Return
        self.on_training_finished(X, y)
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
        start = time.perf_counter()
        # Prepare stratified kfold
        X = self.get_input_from_pcloud(pcloud)
        y = pcloud.get_classes_vector()
        if self.imputer is not None:
            X, y = self.imputer.impute(X, y)
        skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=self.shuffle_points,
            random_state=self.random_seed
        )
        # Compute stratified kfold
        evals = np.full(
            (self.num_folds, len(self.autoval_metrics_names)),
            np.nan
        )
        for i, (Itrain, Ival) in enumerate(skf.split(X, y)):
            Xi, yi = X[Itrain], y[Itrain]
            self.training(Xi, yi, info=False)
            yhat_val = self.predict(None, X=X[Ival])
            evals[i] = self.autoval(y[Ival], yhat_val, info=False)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Stratified kfold training computed in {end-start:.3f} seconds'
        )
        # Report and plot k-fold validation
        kfold_eval = KFoldEvaluator(
            problem_name='Stratified KFold training',
            metric_names=self.autoval_metrics_names
        ).eval(evals)
        if kfold_eval.can_report():
            report = kfold_eval.report()
            LOGGING.LOGGER.info(report)
            if self.stratkfold_report_path is not None:
                report.to_file(path=self.stratkfold_report_path)
        else:
            LOGGING.LOGGER.warning(
                'Model could not report stratified k-folding-based training.'
            )
        if self.stratkfold_plot_path is not None:
            if kfold_eval.can_plot():
                kfold_eval.plot(path=self.stratkfold_plot_path).plot()
            else:
                LOGGING.LOGGER.warning(
                    'Model could not plot stratified k-folding based training.'
                )
        # Return trained model
        self.on_training_finished(X, y)
        return self

    def on_training_finished(self, X, y):
        """
        Callback method that must be invoked by any training strategy after
        finishing the training but before returning the trained model.

        :param X: The point-wise features matrix with points as rows and
            features as columns.
        :param y: The point-wise expected classes.
        :return: Nothing.
        """
        pass

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
