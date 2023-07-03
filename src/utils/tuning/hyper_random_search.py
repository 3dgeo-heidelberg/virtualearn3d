# ---   IMPORTS   --- #
# ------------------- #
from src.utils.tuning.hyper_tuner import HyperTuner


# ---   CLASS   --- #
# ----------------- #
class HyperRandomSearch(HyperTuner):
    """
    :author: Alberto M. Esmoris Pena

    Class to apply random search on the hyperparameter space of a model.
    """
    # TODO Rethink : Implement the class
    pass

    # ---   TUNER METHODS   --- #
    # ------------------------- #
    def tune(self, model, pcloud=None):
        """
        Tune the given model with the best configuration found after computing
        a random search on the model's hyperparameters space.
        See :class:`.HyperTuner` and :class:`.Tuner`.
        Also, see :meth:`tuner.Tuner.tune`
        """
        # TODO Rethink : Implement
        return model
