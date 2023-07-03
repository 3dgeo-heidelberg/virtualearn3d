# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.utils.tuning.tuner import Tuner, TunerException
from src.utils.dict_utils import DictUtils


# ---   CLASS   --- #
# ----------------- #
class HyperTuner(Tuner, ABC):
    """
    :author: Alberto M. Esmoris Pena

    Class for model's hyperparameters tuning.

    :ivar report_path: The path (OPTIONAL) to export the hyperparameter tuning
        report.
    :vartype report_path: str
    :ivar hpnames: The names (as strings) of the hyperparameters to be
        considered.
    :vartype hpnames: list or tuple or np.ndarray
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_tuner_args(spec):
        """
        Extract the arguments to initialize/instantiate an HyperTuner from
        a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an HyperTuner.
        """
        # Initialize
        kwargs = {
            'hyperparameters': spec.get('hyperparameters', None),
            'report_path': spec.get('report_path', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an HyperTuner.

        :param kwargs: The attributes for the HyperTuner.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the HyperTuner
        self.report_path = kwargs.get('report_path', None)
        self.hpnames = kwargs.get('hyperparameters', None)
        if self.hpnames is None:
            raise TunerException(
                'Hyperparameter tuning is not possible without '
                'hyperparameters. None was specified.'
            )
        if len(self.hpnames) < 1:
            raise TunerException(
                'Hyperparameter tuning is not possible without '
                'hyperparameters. Empty set was given.'
            )
