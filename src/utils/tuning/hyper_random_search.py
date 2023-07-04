# ---   IMPORTS   --- #
# ------------------- #
from src.utils.tuning.hyper_tuner import HyperTuner
from src.utils.tuning.tuner import TunerException
from src.utils.dict_utils import DictUtils
from src.report.hyper_search_report import HyperSearchReport
import src.main.main_logger as LOGGING
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as sta
import time


# ---   CLASS   --- #
# ----------------- #
class HyperRandomSearch(HyperTuner):
    """
    :author: Alberto M. Esmoris Pena

    Class to apply random search on the hyperparameter space of a model.

    # TODO Rethink : Document attributes
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_tuner_args(spec):
        """
        Extract the arguments to initialize/instantiate an HyperRandomSearch
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an HyperRandomSearch.
        """
        # Initialize from parent
        kwargs = HyperTuner.extract_tuner_args(spec)
        # Extract particular arguments for random search
        kwargs['nthreads'] = spec.get('nthreads', None)
        kwargs['iterations'] = spec.get('iterations', None)
        kwargs['num_folds'] = spec.get('num_folds', None)
        kwargs['pre_dispatch'] = spec.get('pre_dispatch', None)
        kwargs['distributions'] = spec.get('distributions', None)
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an HyperRandomSearch.

        :param kwargs: The attributes for the HyperRandomSearch
        """
        # Call parent's init
        kwargs = HyperRandomSearch.kwargs_hyperparameters_from_distributions(
            kwargs
        )
        super().__init__(**kwargs)
        # Baic attributes of the HyperRandomSearch
        self.nthreads = kwargs.get('nthreads', -1)
        self.iterations = kwargs.get('iterations', 32)
        self.num_folds = kwargs.get('num_folds', 5)
        self.pre_dispatch = kwargs.get('pre_dispatch', 8)
        self.distributions = kwargs.get('distributions', None)
        if self.distributions is None:
            raise TunerException(
                'No random search on the model\'s hyperparameters is possible '
                'without specifying the distributions.'
            )

    # ---   TUNER METHODS   --- #
    # ------------------------- #
    def tune(self, model, pcloud=None):
        """
        Tune the given model with the best configuration found after computing
        a random search on the model's hyperparameters space.
        See :class:`.HyperTuner` and :class:`.Tuner`.
        Also, see :meth:`tuner.Tuner.tune`

        :param model: The model which hyperparameters must be tuned.
        :param pcloud: The input point cloud (cannot be None)
        """
        # Compute random search
        start = time.perf_counter()
        rs = RandomizedSearchCV(
            model.model,
            self.build_distributions(self.distributions),
            n_iter=self.iterations,
            cv=self.num_folds,
            n_jobs=self.nthreads,
            pre_dispatch=self.pre_dispatch,
            refit=False
        )
        F = pcloud.get_features_matrix(model.fnames)
        y = pcloud.get_classes_vector()
        if model.imputer is not None:
            F, y = model.imputer.impute(F, y)
        rs = rs.fit(F, y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Computed random search in {end-start:.3f} seconds.'
        )
        # Report results
        hsreport = HyperSearchReport(rs.cv_results_)
        LOGGING.LOGGER.info(
            f'Hyperparameter random search report:\n{hsreport}'
        )
        if self.report_path is not None:
            hsreport.to_file(self.report_path)
        # Update model args
        # TODO Rethink : Logic below to HyperTuner.updateModel method
        best_args = rs.best_params_
        best_info = 'Consequences of random search on hyperparameters:'
        for model_arg_key in best_args.keys():
            best_info += '\nModel argument "{arg_name}" ' \
                'from {arg_old} to {arg_new}'.format(
                    arg_name=model_arg_key,
                    arg_old=model.model_args[model_arg_key],
                    arg_new=best_args[model_arg_key]
                )
            model.model_args[model_arg_key] = best_args[model_arg_key]
        best_index = rs.best_index_
        best_info += '\nExpected score with new arguments: '\
            f'{100*rs.best_score_:.3f} '\
            f'+- {100*rs.cv_results_["std_test_score"][best_index]:.3f}\n'\
            f'Expected training time per {len(F)} points with new arguments: '\
            f'{rs.cv_results_["mean_fit_time"][best_index]:.3f} '\
            f'{rs.cv_results_["std_fit_time"][best_index]:.3f}'
        LOGGING.LOGGER.info(best_info)
        # Return tuned model
        return model

    # ---   STATIC UTILS   --- #
    # ------------------------ #
    @staticmethod
    def kwargs_hyperparameters_from_distributions(kwargs):
        """
        Update the key-word arguments (kwargs) to derive the hyperparameters
        from the distributions.

        :param kwargs: The kwargs to be updated
        :return: The updated kwargs
        :rtype: dict
        """
        # TODO Rethink : Implement by calling HyperTuner.kwargs_hyperparameters_from_spec
        hpnames = kwargs.get('hyperparameters', None)
        distros = kwargs.get('distributions', None)
        # Handle cases
        if distros is None:  # If no distributions continue (error later)
            return kwargs
        dnames = [key for key in distros.keys()]  # Distribution keys as pnames
        if hpnames is None:  # If no hyperparameters are given
            # The hyperparameters must be taken from distribution names (keys)
            kwargs['hyperparameters'] = dnames
            return kwargs
        # Both, distributions and hyperparameters are given
        hpnames.sort()  # Sort hyperparameter names
        dnames.sort()  # Sort distribution keys the same way
        hpnames_equals_distros = hpnames == dnames  # Compare sorted lists
        if not hpnames_equals_distros:  # If hyperparams differ from distros
            raise TunerException(
                'HyperRandomSearch received an ambiguous specification. '
                'Hyperparameters and distributions do not match exactly.\n'
                f'Hyperparameters: {hpnames}\n'
                f'Distribution parameters: {dnames}'
            )
        return kwargs

    @staticmethod
    def build_distributions(distributions):
        """
        Transform the specified distributions instantiating the corresponding
        objects to represent random distributions.

        :param distributions: The specification of the distributions.
        :return: The built distributions.
        """
        # Initialize
        distros = {}
        # Populate
        for key in distributions.keys():
            val = distributions[key]
            if isinstance(val, list):  # Assign lists directly
                distros[key] = val
            elif isinstance(val, dict):  # Transform dictionaries
                if val['distribution'] == 'uniform':
                    distros[key] = sta.uniform(
                        loc=val['start'],
                        scale=val['offset']
                    )
                elif val['distribution'] == 'normal':
                    distros[key] = sta.norm(
                        loc=val['mean'],
                        scale=val['stdev']
                    )
                elif val['distribution'] == 'randint':
                    distros[key] = sta.randint(val['start'], val['end'])
                else:
                    raise TunerException(
                        'Unexpected distribution name specified at '
                        f'HyperRandomSearch: "{val["distribution"]}"'
                    )
            else:
                raise TunerException(
                    'Unexpected distribution type specified at '
                    f'HyperRandomSearch with key: "{key}"'
                )
        # Return
        return distros