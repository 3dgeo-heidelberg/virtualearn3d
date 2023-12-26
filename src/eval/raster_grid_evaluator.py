# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.raster_grid_evaluation import RasterGridEvaluation
from src.inout.geotiff_io import GeoTiffIO
from src.utils.dict_utils import DictUtils
from src.utils.str_utils import StrUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
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
    :ivar grids: The many grid specifications.
    :vartype grids: list of dict
    :ivar crs: The coordinate reference system (CRS).
    :vartype crs: str
    :ivar xres: The cell size along the x-axis.
    :vartype xres: float
    :ivar yres: The cell size along the y-axis.
    :vartype yres: float
    :ivar grid_iter_step: How many rows at most must be considered per
        iteration when generating the raster-like grid.
    :vartype grid_iter_step: int
    :ivar reverse_rows: Whether to reverse the rows of the grid (True) or not
        (False).
    :vartype reverse_rows: bool
    :ivar radius_expr: An expression defining the radius of the neighborhood
        centerd on each cell. In this expression, "l" represents the greatest
        cell size, i.e., :math:`\max \; \{\mathrm{xres}, \mathrm{yres}\}`.
    :vartype radius_expr: str
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
            'grids': spec.get('grids', None),
            'crs': spec.get('crs', None),
            'xres': spec.get('xres', None),
            'yres': spec.get('yres', None),
            'grid_iter_step': spec.get('grid_iter_step', None),
            'reverse_rows': spec.get('reverse_rows', None),
            'radius_expr': spec.get('radius_expr', None)
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
        self.grids = kwargs.get('grids', None)
        self.crs = kwargs.get('crs', None)
        self.xres = kwargs.get('xres', None)
        self.yres = kwargs.get('yres', None)
        self.grid_iter_step = kwargs.get('grid_iter_step', 1024)
        self.reverse_rows = kwargs.get('reverse_rows', True)
        self.radius_expr = kwargs.get('radius_expr', 'l')
        # Validate
        if self.grids is None or len(self.grids) < 1:
            raise EvaluatorException(
                'RasterGridEvaluator must be built for at least one grid '
                'specification.'
            )

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, pcloud):
        """
        Evaluate the point cloud as a raster-like 2D grid.

        :param pcloud: The point cloud to be evaluated.
        :return:
        """
        start = time.perf_counter()
        # Extract coordinates
        X = pcloud.get_coordinates_matrix()
        # Compute grids of features
        Fgrids, onames = self.digest_grids(pcloud, X)
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'RasterGridEvaluator evaluated {pcloud.get_num_points()} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return RasterGridEvaluation(
            X=X, Fgrids=Fgrids, onames=onames,
            crs=self.crs, xres=self.xres, yres=self.yres
        )

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

    # ---  FEATURE GRID METHODS  --- #
    # ------------------------------ #
    def digest_grids(self, pcloud, X):
        """
        Generate the grids of features for the requested grid specifications.

        :return: The generated grids of features.
        :rtype: list of :class:`np.ndarray`
        """
        # Prepare spatial grid
        width, height, window, transform, xmin, xmax, ymin, ymax = \
            GeoTiffIO.generate_raster(X, self.xres, self.yres)
        Xgrid = np.array(np.meshgrid(
            np.linspace(xmin, xmax, width),
            np.linspace(ymin, ymax, height)
        )).T
        # Prepare spatial queries
        kdt = KDT(X[:, :2])
        l = max(self.xres, self.yres)  # l is used when evaluating radius_expr
        radius = eval(StrUtils.to_numpy_expr(self.radius_expr))
        # Compute grids of features
        grids = None
        onames = [grid['oname'] for grid in self.grids]
        for i in range(0, width, self.grid_iter_step):  # Iterate over rows
            Xi = np.vstack(Xgrid[i:i+self.grid_iter_step])
            I = KDT(Xi).query_ball_tree(kdt, radius)
            subgrid = []
            for k, grid in enumerate(self.grids):
                subgrid.append(self.digest_grid(
                    pcloud, grid, height, I, n_rows=len(I)//height
                ))
            if grids is None:
                grids = subgrid
            else:
                for j in range(len(grids)):
                    grids[j] = np.vstack([grids[j].T, subgrid[j].T]).T
        # Reverse rows if requested
        for k, grid in enumerate(grids):
            grids[k] = grid[:, ::-1, :]
        # Return
        return grids, onames

    def digest_grid(self, pcloud, grid, height, I, n_rows):
        """
        Generate the grid of features for a given grid specification.

        :param pcloud: The point cloud containing the features.
        :param grid: The grid specification to be digested.
        :param height: The height of the grid in number of cells.
        :param I: The list of neighborhoods, where each neighborhood is
            represented as a list of indices.
        :param n_rows: How many rows are being considered in the chunk to
            be digested.
        :return: The generated grid of features.
        :rtype: :class:`np.ndarray`
        """
        # Obtain features
        fnames = grid['fnames']
        F = pcloud.get_features_matrix(fnames)
        # Determine empty val
        empty_val = grid.get('empty_val', np.nan)
        if isinstance(empty_val, str):
            if empty_val.lower() == 'nan':
                empty_val = np.nan
            else:
                raise EvaluatorException(
                    'RasterGridEvaluator received an unexpected empty value '
                    f'specification: "{empty_val}"'
                )
        empty_val = [empty_val for i in range(F.shape[1])]
        # Determine reduce function
        reduce = grid['reduce']
        reduce_low = reduce.lower()
        reducef = None  # Reduce function
        if reduce_low == 'mean':
            reducef = lambda F, I, j, target, th: np.mean(F[I[j]], axis=0)
        elif reduce_low == 'median':
            reducef = lambda F, I, j, target, th: np.median(F[I[j]], axis=0)
        elif reduce_low == 'min':
            reducef = lambda F, I, j, target, th: np.min(F[I[j]], axis=0)
        elif reduce_low == 'max':
            reducef = lambda F, I, j, target, th: np.max(F[I[j]], axis=0)
        elif reduce_low == 'binary_mask':
            reducef = lambda F, I, j, target, th: [int(np.count_nonzero(
                F[I[j]] == target,
                axis=0
            ) >= th)]
        if reducef is None:
            raise EvaluatorException(
                'RasterGridEvaluator does not support grid digestion without '
                f'a valid reduce function. The requested one \"{reduce}\" is '
                'not acceptable.'
            )
        # Determine target value and threshold
        target = grid.get('target_val', None)
        threshold = grid.get('count_threshold', None)
        # Compute grid of features
        Fgrid = [
            [
                reducef(F, I, i*height+j, target, threshold)
                if len(I[i*height+j]) > 0 else empty_val
                for j in range(height)
            ]
            for i in range(n_rows)
        ]
        return np.array(Fgrid).T

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
