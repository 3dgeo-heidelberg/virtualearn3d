# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import Plot, PlotException
from src.inout.geotiff_io import GeoTiffIO
import src.main.main_logger as LOGGING
import time
import os


# ---   CLASS   --- #
# ----------------- #
class RasterGridPlot(Plot):
    """
    :author: Alberto M. Esmoris Pena

    Class to generate the raster-like plots corresponding to a given point
    cloud.

    See :class:`.Plot` and :class:`.RasterGridEvaluation`.

    TODO Rethink : Doc ivars including vartype
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of RasterGridPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of RasterGridPlot
        self.X = kwargs.get('X', None)
        self.F = kwargs.get('F', None)
        self.crs = kwargs.get('crs', None)
        # TODO Rethink : Assign any pending attribute
        # Validate attributes
        if self.X is None:
            raise PlotException(
                'RasterGridPlot cannot be built without point '
                'coordinates.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot raster-like representations of the point cloud.

        See :meth:`plot.Plot.plot`.
        """
        # Path expansion, if necessary
        _path = self.path
        prefix = kwargs.get('out_prefix', None)
        if prefix is not None:
            prefix = prefix[:-1]  # Ignore '*' at the end
            _path = prefix + _path[1:]
        # Measure time : Start
        start = time.perf_counter()
        # Write GeoTiff
        path = os.path.join(_path, 'general.tiff')
        GeoTiffIO.write(None, path, X=self.X, F=self.F, crs=self.crs)
        # Measure time : End
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'RasterGridPlot generated and wrote figures to '
            f'"{_path}" in {end-start:.3f} seconds.'
        )
