# ---   IMPORTS   --- #
# ⨪------------------ #
from src.model.model import Model, ModelException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.utils.ptransf.fps_decorator_transformer import FPSDecoratorTransformer
from src.main.main_train import MainTrain
import time


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratedModel(Model):
    """
    :author: Alberto M. Esmoris Pena

    Decorator for machine learning models that makes the decorated model work
    on a FPS-based representation of the point cloud.

    The FPS Decorated Model (:class:`.FPSDecoratedModel`)
    constructs a representation of the point cloud, then it calls
    the model on this representation and, when used for predicting, it
    propagates the predictions back to the original point cloud (the one from
    which the representation was built).

    See :class:`.FPSDecoratorTransformer`.

    :ivar decorated_model_spec: The specification of the decorated model.
    :vartype decorated_model_spec: dict
    :ivar decorated_model: The decorated model object.
    :vartype decorated_model: :class:`.Model`
    :ivar undecorated_predictions: Whether to use the decorated model without
        decoration when computing the predictions (true) or not (false).
    :vartype undecorated_predictions: bool
    :ivar fps_decorator_spec: The specification of the FPS transformation
        defining the decorator.
    :vartype fps_decorator_spec: dict
    :ivar fps_decorator: The FPS decorator to be applied on input point clouds.
    :vartype fps_decorator: :class:`.FPSDecoratorTransformer`
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a FPSDecoratedModel
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a FPSDecoratedModel.
        """
        # Initialize from parent
        kwargs = Model.extract_model_args(spec)
        # Extract particular arguments for decorated machine learning models
        kwargs['decorated_model'] = spec.get('decorated_model', None)
        kwargs['fps_decorator'] = spec.get('fps_decorator', None)
        kwargs['undecorated_predictions'] = spec.get(
            'undecorated_predictions', None
        )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type :class:`.FPSDecoratedModel`.
        """
        # Force empty list for fnames in kwargs
        kwargs['fnames'] = []
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the FPSDecoratedModel
        self.decorated_model_spec = kwargs.get('decorated_model', None)
        self.fps_decorator_spec = kwargs.get('fps_decorator', None)
        self.undecorated_predictions = kwargs.get(
            'undecorated_predictions', False
        )
        # Validate decorated model as an egg
        if self.decorated_model_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedModel did not receive any model specification.'
            )
            raise ModelException(
                'FPSDecoratedModel did not receive any model specification.'
            )
        # Validate fps_decorator as an egg
        if self.fps_decorator_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedModel did not receive any decorator '
                'specification.'
            )
            raise ModelException(
                'FPSDecoratedModel did not receive any decorator '
                'specification.'
            )
        # Hatch validated model egg
        model_class = MainTrain.extract_model_class(self.decorated_model_spec)
        self.decorated_model = MainTrain.extract_pretrained_model(
            self.decorated_model_spec,
            expected_class=model_class
        )
        if self.decorated_model is None:  # Initialize model when no pretrained
            self.decorated_model = model_class(
                **model_class.extract_model_args(self.decorated_model_spec)
            )
        else:
            self.decorated_model.overwrite_pretrained_model(
                model_class.extract_model_args(self.decorated_model_spec)
            )
        # Hatch validated fps_decorator egg
        self.fps_decorator = FPSDecoratorTransformer(**self.fps_decorator_spec)
        # Warn about not recommended number of encoding neighbors
        nen = self.fps_decorator_spec.get('num_encoding_neighbors', None)
        if nen is None or nen != 1:
            LOGGING.LOGGER.warning(
                f'FPSDecoratedModel received {nen} encoding neighbors. '
                'Using a value different than one is not recommended as it '
                'could easily lead to unexpected behaviors.'
            )

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def train(self, pcloud):
        """
        Decorate the main training logic to work on the representation.
        See :class:`.Model` and :meth:`.Model.train`.
        """
        # Build representation from input point cloud
        fnames = self.get_fnames_recursively()
        start = time.perf_counter()
        rf_pcloud = self.fps_decorator.transform_pcloud(pcloud, fnames=fnames)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'FPSDecoratedModel computed a representation of '
            f'{pcloud.get_num_points()} points using '
            f'{rf_pcloud.get_num_points()} points '
            f'for training in {end-start:.3f} seconds.'
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Train the model
        start = time.perf_counter()
        training_output = self.decorated_model.train(rf_pcloud)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'{self.decorated_model.__class__.__name__} trained on the '
            f'representation space in {end-start:3f} seconds.'
        )
        return training_output

    def predict(self, pcloud, X=None):
        """
        Decorate the main predictive logic to work on the representation.
        See :class:`.Model` and :meth:`.Model.predict`.
        """
        # Delegate on decorated model if undecorated predictions are requested
        if self.undecorated_predictions:
            start = time.perf_counter()
            yhat = self.decorated_model.predict(pcloud, X=X)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'FPSDecoratedModel computed undecorated predictions on '
                f'{len(yhat)} points in {end-start:.3f} seconds.'
            )
            return yhat
        # Build representation from input point cloud
        fnames = self.get_fnames_recursively()
        start = time.perf_counter()
        rf_pcloud = self.fps_decorator.transform_pcloud(pcloud, fnames=fnames)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'FPSDecoratedModel computed a representation of '
            f'{pcloud.get_num_points()} points using '
            f'{rf_pcloud.get_num_points()} points '
            f'for predictions in {end-start:.3f} seconds.'
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Predict
        start = time.perf_counter()
        rf_yhat = self.decorated_model.predict(rf_pcloud)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'{self.decorated_model.__class__.__name__} predicted on the '
            f'representation space in {end-start:.3f} seconds.'
        )
        # Delete rf_pcloud
        rf_pcloud = None
        # Propagate predictions to original point cloud and return
        start = time.perf_counter()
        yhat = self.fps_decorator.propagate(rf_yhat)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'FPSDecoratedModel propagated predictions in '
            f'{end-start:.3f} seconds.'
        )
        return yhat

    def prepare_model(self):
        """
        Prepare the decorated model.
        See :class:`.Model` and :meth:`.Model.prepare_model`.
        """
        self.decorated_model.prepare_model()

    def overwrite_pretrained_model(self, spec):
        """
        Overwrite the decorated pretrained model.
        See :class:`.Model` and :meth:`.Model.overwrite_pretrained_model`.
        """
        self.decorated_model.overwrite_pretrained_model(
            spec["decorated_model_spec"]
        )

    def get_input_from_pcloud(self, pcloud):
        """
        Get input from the decorated pretrained model.
        See :class:`.Model` and :meth:`.Model.get_input_from_pcloud`.
        """
        return self.decorated_model.get_input_from_pcloud(pcloud)

    def is_deep_learning_model(self):
        """
        Check whether the decorated model is a deep learning model.
        See :class:`.Model` and :meth:`.Model.is_deep_learning_model`.
        """
        return self.decorated_model.is_deep_learning_model()

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y, info=True):
        """
        Use the training logic of the decorated model.
        See :class:`.Model` and :meth:`.Model.training`.
        """
        return self.decorated_model.training(X, y, info=info)

    def autoval(self, y, yhat, info=True):
        """
        Auto validation during training through decorated model.
        See :class:`.Model` and :meth:`.Model.autoval`.
        """
        return self.decorated_model.autoval(y, yhat, info=info)

    def train_base(self, pcloud):
        """
        Straightforward training through decorated model.
        See :class:`.Model` and :meth:`.Model.train_base`.
        """
        return self.decorated_model.train_base(pcloud)

    def train_autoval(self, pcloud):
        """
        Use autovalidation training strategy from decorated model.
        See :class:`.Model` and :meth:`.Model.train_autoval`.
        """
        return self.decorated_model.train_autoval(pcloud)

    def train_stratified_kfold(self, pcloud):
        """
        Use stratified k-fold training strategy from decorated model.
        See :class:`.Model` and :meth:`.Model.train_stratified_kfold`.
        """
        return self.decorated_model.train_stratified_kfold(pcloud)

    def on_training_finished(self, X, y):
        """
        Use on training finished callback from decorated model.
        See :class:`.Model` and :meth:`.Model.on_training_finished`.
        """
        self.decorated_model.on_training_finished(X, y)

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, **kwargs):
        """
        Do the predictions through the decorated model.
        See :class:`.Model` and :meth:`.Model._predict`.
        """
        return self.decorated_model._predict(X, **kwargs)

    # ---  FPS DECORATED MODEL METHODS  --- #
    # ------------------------------------- #
    def get_fnames_recursively(self):
        """
        Find through any potential decoration graph until the deepest model
        is found, then consider its feature names.

        :return: The feature names of the deepest model in the decoration
            hierarchy.
        :rtype: list of str
        """
        # Find through all decorated models until the last one (deepest)
        submodel = self.decorated_model
        while hasattr(submodel, 'decorated_model'):
            submodel = submodel.decorated_model
        # Return the feature names of the deepest model
        return submodel.fnames

    @property
    def fnames(self):
        """
        Getter for feature names to get them from the decorated model.
        :return: The feature names from the decorated model.
        :rtype: None or list of str
        """
        if hasattr(self, 'decorated_model'):
            return self.decorated_model.fnames
        return []  # Return empty list instead of None if no fnames

    @fnames.setter
    def fnames(self, fnames):
        """
        Setter for feature names to set them in the decorated model.
        :param fnames: The feature names for the decorated model.
        :type fnames: None or list of str
        """
        if hasattr(self, 'decorated_model'):
            self.decorated_model.fnames = fnames
