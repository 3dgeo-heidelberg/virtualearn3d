# ---   IMPORTS   --- #
# ------------------- #
from src.utils.tuning.hyper_tuner import HyperTuner
from src.utils.tuning.tuner import TunerException
from sklearn.model_selection import GridSearchCV
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.report.hyper_search_report import HyperSearchReport
import time


# ---   CLASS   --- #
# ----------------- #
class HyperGridSearch(HyperTuner):
    """
    :author: Alberto M. Esmoris Pena

    Class to apply grid search on the hyperparameter space of a model.

    :ivar nthreads: Number of threads to run parallel grid search nodes.
        Note that the model might have nthreads too. In this case, it is
        important that the number of threads from the grid search and the
        number of threads from the model itself are considered together.
    :vartype nthreads: int
    :ivar num_folds: Number of folds to train and validate the model on a
        kfolding scheme for each node in the grid search.
    :vartype num_folds: int
    :ivar pre_dispatch: How many jobs are dispatched during the parallel
        execution. It can be useful to prevent dispatching more jobs than
        those that can be processed by the CPUs.
    :vartype pre_dispatch: int or str
    :ivar grid: Dictionary which elements are lists. Each list in the
        dictionary represents the values that must be searched for the
        parameter referenced by the key.
    :vartype grid: dict
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_tuner_args(spec):
        """
        Extract the arguments to initialize/instantiate an HyperGridSearch
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an HyperGridSearch.
        """
        # Initialize from parent
        kwargs = HyperTuner.extract_tuner_args(spec)
        # Extract particular arguments for grid search
        kwargs['nthreads'] = spec.get('nthreads', None)
        kwargs['num_folds'] = spec.get('num_folds', None)
        kwargs['pre_dispatch'] = spec.get('pre_dispatch', None)
        kwargs['grid'] = spec.get('grid', None)
        # Delete keys with None value
        DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an HyperGridSearch.

        :param kwargs: The attributes for the HyperGridSearch
        """
        # Call parent's init
        kwargs = HyperGridSearch.kwargs_hyperparameters_from_grid(kwargs)
        super().__init__(**kwargs)
        # Basic attributes of the HyperGridSearch
        self.nthreads = kwargs.get('nthreads', -1)
        self.num_folds = kwargs.get('num_folds', 5)
        self.pre_dispatch = kwargs.get('pre_dispatch', 8)
        self.grid = kwargs.get('grid', None)
        if self.grid is None:
            raise TunerException(
                'No grid search on the model\'s hyperparameters is possible '
                'without a grid specification.'
            )

    # ---   TUNER METHODS   --- #
    # ------------------------- #
    def tune(self, model, pcloud=None):
        """
        Tune the given model with the best configuration found after computing
        a grid search on the model's hyperparameters space.
        See :class:`.HyperTuner` and :class:`.Tuner`.
        Also, see :meth:`tuner.Tuner.tune`

        :param model: The model which hyperparameters must be tuned.
        :param pcloud: The input point cloud (cannot be None).
        """
        # Compute grid search
        start = time.perf_counter()
        gs = GridSearchCV(
            model.model,
            self.grid,
            cv=self.num_folds,
            n_jobs=self.nthreads,
            pre_dispatch=self.pre_dispatch,
            refit=False
        )
        F = pcloud.get_features_matrix(model.fnames)
        y = pcloud.get_classes_vector()
        if model.imputer is not None:
            F, y = model.imputer.impute(F, y)
        gs = gs.fit(F, y)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Computed grid search in {end-start:.3f} seconds.'
        )
        # Report results
        hsreport = HyperSearchReport(gs.cv_results_)
        LOGGING.LOGGER.info(
            f'Hyperparameter grid search report:\n{hsreport}'
        )
        if self.report_path is not None:
            hsreport.to_file(self.report_path)
        # Update model args
        # TODO Rethink : Logic below to HyperTuner.updateModel method
        best_args = gs.best_params_
        best_info = 'Consequences of grid search on hyperparameters:'
        for model_arg_key in best_args.keys():
            best_info += '\nModel argument "{arg_name}" ' \
                'from {arg_old} to {arg_new}'.format(
                    arg_name=model_arg_key,
                    arg_old=model.model_args[model_arg_key],
                    arg_new=best_args[model_arg_key]
                )
            model.model_args[model_arg_key] = best_args[model_arg_key]
        best_index = gs.best_index_
        best_info += '\nExpected score with new arguments: '\
            f'{100*gs.best_score_:.3f} '\
            f'+- {100*gs.cv_results_["std_test_score"][best_index]:.3f}\n'\
            f'Expected training time per {len(F)} points with new arguments: '\
            f'{gs.cv_results_["mean_fit_time"][best_index]:.3f} '\
            f'+- {gs.cv_results_["std_fit_time"][best_index]:.3f}'
        LOGGING.LOGGER.info(best_info)
        # Return tuned model
        return model

    # ---   STATIC UTILS   --- #
    # ------------------------ #
    @staticmethod
    def kwargs_hyperparameters_from_grid(kwargs):
        """
        Update the key-word arguments (kwargs) to derive the hyperparameters
        from the grid.

        :param kwargs: The kwargs to be updated.
        :return: The updated kwargs.
        :rtype: dict
        """
        # TODO Rethink : Implement by calling HyperTuner.kwargs_hyperparameters_from_spec
        hpnames = kwargs.get('hyperparameters', None)
        grid = kwargs.get('grid', None)
        # Handle cases
        if grid is None:  # If no grid continue (error will arise later)
            return kwargs
        gnames = [key for key in grid.keys()]  # Grid keys as parameter names
        if hpnames is None:  # If grid is given but no hyperparameters
            # The hyperparameters must be taken from grid names (keys)
            kwargs['hyperparameters'] = gnames
            return kwargs
        # Both, grid and hyperparameters are given
        hpnames.sort()  # Sort hyperparameter's names
        gnames.sort()  # Sort grid keys the same way
        hpnames_equals_grid = hpnames == gnames  # Compare sorted lists
        if not hpnames_equals_grid:  # If hyperparameters differ from grid
            raise TunerException(
                'HyperGridSearch received an ambiguous specification. '
                'Hyperparameters and grid do not match exactly.\n'
                f'Hyperparameters: {hpnames}\n'
                f'Grid parameters: {gnames}'
            )
        return kwargs
