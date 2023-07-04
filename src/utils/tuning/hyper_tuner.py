# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.utils.tuning.tuner import Tuner, TunerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING


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

    # ---  HYPERTUNER METHODS  --- #
    # ---------------------------- #
    def update_model(self, model, search, features=None):
        """
        Update model from result.

        :param model: The model to be updated. See :class:`.Model`
        :param search: The search to update the model.
        :param features: The features matrix (OPTIONAL, i.e., can be None).
        :return: The updated model.
        :rtype: :class:`.Model`
        """
        # Extract from search and features
        best_args = search.best_params_
        best_index = search.best_index_
        best_score = search.best_score_
        results = search.cv_results_
        num_points = len(features) if features is not None else '?'
        # Update model (and build log message)
        best_info = 'Consequences of random search on hyperparameters:'
        for model_arg_key in best_args.keys():
            best_info += '\nModel argument "{arg_name}" ' \
                         'from {arg_old} to {arg_new}'.format(
                arg_name=model_arg_key,
                arg_old=model.model_args[model_arg_key],
                arg_new=best_args[model_arg_key]
            )
            model.model_args[model_arg_key] = best_args[model_arg_key]
        best_info += '\nExpected score with new arguments: ' \
            f'{100*best_score:.3f} ' \
            f'+- {100*results["std_test_score"][best_index]:.3f}\n' \
            f'Expected training time per {num_points} points ' \
            'with new arguments: ' \
            f'{results["mean_fit_time"][best_index]:.3f} ' \
            f'+-{results["std_fit_time"][best_index]:.3f} seconds'
        LOGGING.LOGGER.info(best_info)
        # Return updated model
        return model

    # ---   STATIC UTILS   --- #
    # ------------------------ #
    @staticmethod
    def kwargs_hyperparameters_from_spec(kwargs, spec):
        # TODO Rethink : Implement (see grid and random search static utils)
        pass
