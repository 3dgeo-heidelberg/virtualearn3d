# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.utils.tuning.tuner import Tuner, TunerException
from src.utils.dict_utils import DictUtils
from sklearn.utils import shuffle
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
    @staticmethod
    def search(model, search, pcloud):
        """
        Compute the search of the best hyperparameters for the given model
        on the given point cloud.

        :param model: The model which hyperparameters must be tuned. See
            :class:`.Model`.
        :param search: The search object (must have a fit method to be applied
            on a features matrix and a vector of classes, i.e., F and y).
        :param pcloud: The point cloud representing the input data for the
            search.
        :return: Completed search.
        """
        F = pcloud.get_features_matrix(model.fnames)
        y = pcloud.get_classes_vector()
        if model.imputer is not None:
            F, y = model.imputer.impute(F, y)
        if model.shuffle_points:
            F, y = shuffle(F, y)
        return search.fit(F, y)

    @staticmethod
    def update_model(model, search, pcloud=None):
        """
        Update model from result.

        :param model: The model to be updated. See :class:`.Model`
        :param search: The search to update the model.
        :param features: The input point cloud (OPTIONAL, i.e., can be None).
        :return: The updated model.
        :rtype: :class:`.Model`
        """
        # Extract from search and features
        best_args = search.best_params_
        best_index = search.best_index_
        best_score = search.best_score_
        results = search.cv_results_
        num_points = pcloud.get_num_points() if pcloud is not None else '?'
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
            f'+- {results["std_fit_time"][best_index]:.3f} seconds'
        LOGGING.LOGGER.info(best_info)
        # Return updated model
        return model

    # ---   STATIC UTILS   --- #
    # ------------------------ #
    @staticmethod
    def kwargs_hyperparameters_from_spec(kwargs, spec):
        """
        Update the key-word arguments (kwargs) to derive the hyperparameters
        from the specification. In case there are explicitly given
        hyperparameters, they must match exactly with the specification.

        :param kwargs: The key-word arguments to be updated.
        :param spec: The specification, often a dictionary contained inside the
            key-word arguments.
        :return: The updated key-word arguments.
        """
        # Extract the name of  the hyperparameters
        hpnames = kwargs.get('hyperparameters', None)
        # Handle cases
        if spec is None:  # If no specification, then continue (error later)
            return kwargs
        snames = [key for key in spec.keys()]  # Spec. keys as param. names
        if hpnames is None:  # If no hyperparameters are given
            # The hyperparameters must be taken from spec. names (keys)
            kwargs['hyperparameters'] = snames
            return kwargs
        # Both, specification and hyperparameters are given
        hpnames.sort()  # Sort hyperparameter names
        snames.sort()  # Sort specification's keys the same way
        hpnames_equals_snames = hpnames == snames  # Compare sorted lists
        if not hpnames_equals_snames:  # If hyperparams differ from spec.
            raise TunerException(
                'HyperTuner received an ambiguous specification. '
                'Hyperparameters and specification do not match exactly.\n'
                f'Hyperparameters: {hpnames}\n'
                f'Specified parameters: {snames}'
            )
        return kwargs
