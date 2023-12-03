# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.raster_grid_evaluation import RasterGridEvaluation
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class RasterGridEvaluator(Evaluator):
    r"""
    :author: Alberto M. Esmoris Pena

    Class to generate a raster-like 2D grid evaluating a given point cloud.

    :ivar plot_path: The path to write the raster.
    :vartype plot_path: str
    :ivar fnames: The name of the features to be considered.
    :vartype fnames: list of str

    TODO Rethink : Doc ivars including vartype
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_eval_args(spec):
        """
        Extract the arguments to initialize/instantiate a

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            RasterGridEvaluator.
        :rtype: dict
        """
        # Initialize
        kwargs = {
            'plot_path': spec.get('plot_path', None),
            'fnames': spec.get('fnames', None),
            'crs': spec.get('crs', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a RasterGridEvaluator.

        :param kwargs: The attributes for the RasterGridEvaluator.
        """
        # Call parent''s init
        kwargs['problem_name'] = 'RASTER_GRID'
        super().__init__(**kwargs)
        # Assign RasterGridEvaluator attributes
        self.plot_path = kwargs.get('plot_path', None)
        self.fnames = kwargs.get('fnames', None)
        self.crs = kwargs.get('crs', None)
        # TODO Rethink : Assign any pending attribute

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, pcloud):
        """
        Evaluate the point cloud as a raster-like 2D grid.

        :param pcloud: The point cloud to be evaluated.
        :return:
        """
        start = time.perf_counter()
        # Extract coordinates and features
        X = pcloud.get_coordinates_matrix()
        fnames = self.fnames
        if fnames is None:
            fnames = pcloud.get_features_names()
        F = pcloud.get_features_matrix(self.fnames)
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'RasterGridEvaluator evaluated {pcloud.get_num_points()} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return RasterGridEvaluation(X=X, F=F, crs=self.crs)

    def __call__(self, pcloud, **kwargs):
        """
        Evaluate with extra logic that is convenient for pipeline-based
        execution.

        See :meth:`evaluator.Evaluator.eval`.

        :param pcloud: The point cloud that must be evaluated through
            raster-like grid analysis.
        """
        # Obtain evaluation
        ev = self.eval(pcloud)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_plot() and self.plot_path is not None:
            start = time.perf_counter()
            ev.plot(path=self.plot_path).plot(out_prefix=out_prefix)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'The RasterGridEvaluator wrote the raster-like plots '
                f'in {end-start:.3f} seconds.'
            )

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def eval_args_from_state(self, state):
        """
        Obtain the arguments to call the RasterGridEvaluator form the current
        pipeline's state.

        :param state: The pipeline's state.
        :type state: :class:`.SimplePipelineState`
        :return: The dictionary of arguments for calling
            RasterGridEvaluator
        :rtype: dict
        """
        return {
            'pcloud': state.pcloud
        }
