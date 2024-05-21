# ---   IMPORTS   --- #
# ------------------- #
from src.model.tdcomp.training_data_component import TrainingDataComponent
import src.main.main_logger as LOGGING
from imblearn.over_sampling import SMOTE as _SMOTE
import time


# ---   CLASS   --- #
# ----------------- #
class SMOTE(TrainingDataComponent):
    """
    :author: Alberto M. Esmoris Pena

    Training data component based on Synthetic Minority Oversampling
    TEchnique (SMOTE). It transforms the input data by considering the
    k-nearest neighbors for each point and interpolates between them to
    generate new samples.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the SMOTE training data component.

        :param kwargs: The attributes for the SMOTE training data component.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   CALL   --- #
    # ---------------- #
    def __call__(self, X, y):
        """
        Apply the SMOTE to transform the input training data.

        See :meth:`.TrainingDataComponent.__call__`.
        """
        m = X.shape[0]
        start = time.perf_counter()
        X, y = _SMOTE(**self.component_args).fit_resample(X, y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'SMOTE applied to {m} training data points yielded '
            f'{X.shape[0]} training data points in '
            f'{end-start:.3f} seconds.'
        )
        return X, y
