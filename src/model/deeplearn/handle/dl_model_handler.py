# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.model.deeplearn.deep_learning_exception import DeepLearningException


# ---   CLASS   --- #
# ----------------- #
class DLModelHandler:
    """
    Abstract class to handle deep learning models. Typically, fitting,
    predicting, and compiling are the main operations supported by deep
    learning model handlers.

    :ivar arch: The model's architecture.
    :vartype arch: :class:`.Architecture`
    :ivar: compilation_args: The key-word specification on how to compile
        the model.
    :vartype compilation_args: dict
    :ivar class_names: The name for each class involved in the classification
        problem, if any (it can be ignored by regression models).
    :vartype class_names: list
    :ivar compiled: It is None by default, but it will be assigned the compiled
        model after calling the fit or predict methods.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, arch, **kwargs):
        """
        Initialize/instantiate a deep learning model handler.

        :param arch: The architecture to be handled.
        :type arch: :class:`.Architecture`
        :param kwargs: The key-word specification to instantiate the
            DLModelHandler.
        """
        # Assign member attributes
        self.arch = arch
        self.compilation_args = kwargs.get('compilation_args', None)
        self.class_weight = kwargs.get('class_weight', None)
        self.class_names = kwargs.get('class_names', None)
        self.compiled = None

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def fit(self, X, y, F=None):
        """
        Fit the handled model to given data.

        :param X: The structure space matrix, typically the matrix with the
            x, y, z coordinates as columns.
        :type X: :class:`np.ndarray`
        :param y: The vector of expected labels, the ground-truth from the
            supervised training perspective.
        :type y: :class:`np.ndarray`
        :param F: The features matrix. Often, models can work without features
            because they can derive their own features from the X matrix.
        :type F: :class:`np.ndarray`
        :return: The fit model handler.
        :rtype: :class:`.DLModelHandler`
        """
        if not self.is_compiled():
            self.compile(X=X, F=F, y=y, arch_plot=True)
        return self._fit(X, y, F=F)

    @abstractmethod
    def _fit(self, X, y, F=None):
        """
        This method must be overriden by any concrete derived class to provide
        the fit logic assuming the model has been compiled. It complements the
        :meth:`dl_model_handler.DLModelHandler.fit` method.

        :return: The fit model handler.
        :rtype: :class:`.DLModelHandler`
        """
        pass

    def predict(self, X, F=None, y=None, zout=None, plots_and_reports=True):
        """
        Compute predictions for the given input data.

        :param X: The structure space matrix, typically the matrix with the
            x, y, z coordinates as columns.
        :type X: :class:`np.ndarray`
        :param F: The features matrix. Often, models can work without features
            because they can derive their own features from the X matrix.
        :type F: :class:`np.ndarray`
        :param y: The vector of expected labels, the ground-truth from the
            supervised training perspective. While it is not necessary to
            compute predictions, when available it can be given because some
            models can use it for evaluation and analysis purposes.
        :type y: :class:`np.ndarray`
        :param zout: It can be given as an empty list in which case its last
            element after the call will contain the output from the last layer
            of the neural network, e.g., the softmax scores for a point-wise
            classification neural network. It can be None, in which case the
            output from the last layer will not be considered. Note also that
            zout does not necessarily return the softmax output, it can be
            defined to consider different output layers or metrics for some
            potential model.
        :param plots_and_reports: Control whether to compute and export the
            plots and reports associated to the computation of predictions
            (True) or not (False).
        :type plots_and_reports: bool
        :return: The predictions.
        :rtype: :class:`np.ndarray`
        """
        if not self.is_compiled():
            self.compile(X=X, F=F)
        return self._predict(
            X, F=F, y=y, zout=zout, plots_and_reports=plots_and_reports
        )

    @abstractmethod
    def _predict(self, X, F=None, y=None, zout=None, plots_and_reports=True):
        """
        This method must be overriden by any concrete derived class to provide
        the predictive logic assuming the model has been compiled. It
        complements the :meth:`dl_model_handler.DLModelHandler.predict` method.

        :return: The predictions.
        :rtype: :class:`np.ndarray`
        """
        pass

    @abstractmethod
    def compile(self, X=None, y=None, F=None, **kwargs):
        """
        The method that provides the logic to compile a model.

        :param X: Optionally, the coordinates might be used for a better
            initialization (e.g., automatically derive the number of expected
            input points, or the dimensionality of the space where the points
            belong to).
        :param y: Optionally, the labels might be used for a better
            initialization (e.g., automatically derive the number of classes).
        :param F: Optionally, the features might be used (or even necessary)
            for initialization (e.g., automatically deriving the dimensionality
            of the input feature space).
        :return: The model handler itself after compiling the architecture,
            which implies modifying its internal state.
        :rtype: :class:`.DLModelHandler`
        """
        pass

    def is_compiled(self):
        """
        Check whether the handled model has been compiled (True) or not
        (False).

        :return: True if the handled model has been compiled, False otherwise.
        :rtype: bool
        """
        return self.compiled is not None

    def overwrite_pretrained_model(self, spec):
        """
        Assist the :meth:`model.Model.overwrite_pretrained_model` method for
        deep learning models.

        :param spec: The key-word specification containing the model's
            arguments.
        :type spec: dict
        """
        spec_keys = spec.keys()
        # Overwrite baseline attributes of the deep learning model handler
        if 'model_handling' in spec_keys:
            spec_handling = spec['model_handling']
            spec_handling_keys = spec_handling.keys()
            if 'class_weight' in spec_handling_keys:
                self.class_weight = spec_handling['class_weight']
            if 'class_names' in spec_handling_keys:
                self.class_names = spec_handling['class_names']
        # Overwrite compilation arguments
        if 'compilation_args' in spec_keys:
            self.compilation_args = spec['compilation_args']
        # Overwrite the attributes of the model's architecture
        if self.arch is not None:
            self.arch.overwrite_pretrained_model(spec)

    # ---  MODEL HANDLING TASKS  --- #
    # ------------------------------ #
    def build_callbacks(self):
        """
        Build the callbacks for the model.

        By default, the abstract baseline DLModelHandler does not provide an
        implementation to build callbacks. Derived classes that need to work
        with model callbacks must override this method to implement the
        building of any necessary callback.

        :return: List of built callbacks
        :rtype: list
        """
        raise DeepLearningException(
            'DLModelHandler does not support building callbacks.'
        )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized deep learning model
        handler.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Return DL Model Handler state (for serialization)
        return {  # Must not include compiled model (it will be rebuilt)
            'arch': self.arch,
            'compilation_args': self.compilation_args,
            'class_weight': self.class_weight,
            'class_names': self.class_names,
            'compiled': None  # Compiled model is not serialized
        }

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized deep learning model handler.

        :param state: The state's dictionary of the saved deep learning model
            handler.
        :return: Nothing, but modifies the internal state of the object.
        """
        # Must rebuild the compiled model (it was not serialized)
        self.arch = state['arch']
        self.compilation_args = state['compilation_args']
        self.class_weight = state['class_weight']
        self.class_names = state['class_names']
        self.compiled = None
