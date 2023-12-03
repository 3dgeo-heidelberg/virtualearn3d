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
        self.Fgrids = kwargs.get('Fgrids', None)
        self.onames = kwargs.get('onames', None)
        self.crs = kwargs.get('crs', None)
        self.xres = kwargs.get('xres', None)
        self.yres = kwargs.get('yres', None)
        # TODO Rethink : Assign any pending attribute
        # Validate attributes
        if self.X is None:
            raise PlotException(
                'RasterGridPlot cannot be built without point '
                'coordinates.'
            )
        if self.Fgrids is None:
            raise PlotException(
                'RasterGridPlot cannot be built without grids of features.'
            )
        if self.onames is None:
            raise PlotException(
                'RasterGridPlot cannot be built without output names.'
            )
        if len(self.Fgrids) != len(self.onames):
            raise PlotException(
                f'RasterGridPlot received {len(self.Fgrids)} grids of '
                f'features and {len(self.onames)} output names. '
                'These numbers MUST be equal but they are not.'
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
        # Handle many paths from onames
        paths = [_path+oname+".tiff" for oname in self.onames]
        # Write GeoTiffs
        for i, path in enumerate(paths):  # For each path
            # Write GeoTiff
            GeoTiffIO.write(
                (self.X, self.Fgrids[i]), path,
                crs=self.crs, xres=self.xres, yres=self.yres
            )
        # Measure time : End
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'RasterGridPlot generated and wrote figures to '
            f'"{_path}" in {end-start:.3f} seconds.'
        )
