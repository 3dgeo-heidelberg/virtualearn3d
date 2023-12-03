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
        self.F = kwargs.get('F', None)
        self.crs = kwargs.get('crs', None)
        # TODO Rethink : Assign any pending attribute
        # Validate attributes
        if self.X is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot be built without point '
                'coordinates.'
            )

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def can_plot(self, **kwargs):
        """
        See :class:`.Evaluation` and :meth:`evaluation.Evaluation.can_plot`.
        """
        return (
            self.X is not None and
            self.F is not None
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
        path = kwargs.get('path', None)
        if path is None:
            raise EvaluationException(
                'RasterGridEvaluation cannot plot without a path.'
            )
        return RasterGridPlot(
            X=self.X,
            F=self.F,
            crs=self.crs,
            path=path
        )
