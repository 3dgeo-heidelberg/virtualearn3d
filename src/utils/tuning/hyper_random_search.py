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
    :ivar nthreads: Number of threads to run parallel grid search nodes.
        Note that the model might have nthreads too. In this case, it is
        important that the number of threads from the grid search and the
        number of threads from the model itself are considered together.
    :vartype nthreads: int
    :ivar num_folds: Number of folds to train and validate the model on a
        kfolding scheme for each node in the grid search.
    :vartype num_folds: int
    :ivar iterations: The number of iterations. Each iteration will randomly
        select values from the random distributions to test a model.
    :vartype iterations: int
    :ivar distributions: Dictionary which elements are dictionaries that define
        a particular distribution to generate random values for a concrete
        hyperparameter.
    :vartype distributions: dict
    :ivar pre_dispatch: How many jobs are dispatched during the parallel
        execution. It can be useful to prevent dispatching more jobs than
        those that can be processed by the CPUs.
    :vartype pre_dispatch: int or str
    :ivar num_folds:
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
        kwargs = HyperTuner.kwargs_hyperparameters_from_spec(
            kwargs,
            kwargs.get('distributions', None)
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
        rs = HyperTuner.search(model, rs, pcloud)
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
        # Update model args and return tuned model
        return self.update_model(model, rs, pcloud)

    # ---   STATIC UTILS   --- #
    # ------------------------ #
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
