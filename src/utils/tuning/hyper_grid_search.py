# ---   IMPORTS   --- #
# ------------------- #
from src.utils.tuning.hyper_tuner import HyperTuner
from src.utils.tuning.tuner import TunerException
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold  # TODO Remove
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
        kwargs = HyperTuner.kwargs_hyperparameters_from_spec(
            kwargs,
            kwargs.get('grid', None)
        )
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
        gs = HyperTuner.search(model, gs, pcloud)
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
        # Update model args and return tuned model
        return self.update_model(model, gs, pcloud)
