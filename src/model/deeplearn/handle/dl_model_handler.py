# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class DLModelHandler:
    """
    # TODO Rethink : Document class and ivars
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
        self.compiled = None

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def fit(self, X, y, F=None):
        # TODO Rethink : Sphinx doc
        if not self.is_compiled():
            self.compile(X=X, F=F, y=y)
        return self._fit(X, y, F=F)

    @abstractmethod
    def _fit(self, X, y, F=None):
        # TODO Rethink : Sphinx doc
        pass

    def predict(self, X, F=None):
        # TODO Rethink : Sphinx doc
        if not self.is_compiled():
            self.compile(X=X, F=F)
        return self._predict(X, F=F)

    @abstractmethod
    def _predict(self, X, F=None):
        # TODO Rethink : Sphinx doc
        pass

    @abstractmethod
    def compile(self, X=None, y=None, F=None):
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
        return self.compiled is not None

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        # TODO Rethink : Sphinx doc
        # Return DL Model Handler state (for serialization)
        return {  # Must not include compiled model (it will be rebuilt)
            'arch': self.arch,
            'compilation_args': self.compilation_args,
            'compiled': None  # Compiled model is not serialized
        }

    def __setstate__(self, state):
        # TODO Rethink : Sphinx doc
        # Must rebuild the compiled model (it was not serialized)
        self.arch = state['arch']
        self.compilation_args = state['compilation_args']
        self.compile()
