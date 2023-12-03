# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation, EvaluationException
from src.plot.raster_grid_plot import RasterGridPlot


# ---   CLASS   --- #
# ----------------- #
class RasterGridEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a point cloud by transforming
    it to a convenient raster representation.

    TODO Rethink : Doc ivars including vartype
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a RasterGridEvaluation.

        :param kwargs: The attributes for the RasterGridEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of RasterGridEvaluation
        self.X = kwargs.get('X', None)
        self.Fgrids = kwargs.get('Fgrids', None)
        self.onames = kwargs.get('onames', None)
        self.crs = kwargs.get('crs', None)
        self.xres = kwargs.get('xres', None)
        self.yres = kwargs.get('yres', None)
        # Validate attributes
        if self.X is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot be built without point '
                'coordinates.'
            )
        if self.Fgrids is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot be built without grids of '
                'features.'
            )
        if self.onames is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot be built without output '
                'names.'
            )

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def can_plot(self, **kwargs):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`.
        """
        return (
            self.X is not None and
            self.Fgrids is not None and
            self.onames is not None
        )

    def plot(self, **kwargs):
        """
        Transform the evaluation into a plot (or many plots), each representing
        a raster-like 2D grid, typically in GeoTiff format.

        See :class:`.RasterGridPlot`.

        :param kwargs: The key-word arguments for the plot.
        :return: The RasterGridPlot representing the RasterGridEvaluation.
        :rtype: :class:`.RasterGridPlot`
        """
        # Handle paths from base path and onames
        path = kwargs.get('path', None)
        if path is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot plot without a path.'
            )
        return RasterGridPlot(
            X=self.X,
            Fgrids=self.Fgrids,
            onames=self.onames,
            crs=self.crs,
            xres=self.xres,
            yres=self.yres,
            path=path
        )
