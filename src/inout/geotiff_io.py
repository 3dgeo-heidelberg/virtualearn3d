# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud import PointCloud
import src.main.main_logger as LOGGING
from scipy.spatial.kdtree import KDTree as KDT
import rasterio
import numpy as np
import os
import time


# ---   CLASS   --- #
# ----------------- #
class GeoTiffIO:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for input/output operations related to
    GeoTiff files.
    """
    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def write(x, path, **kwargs):
        # TODO Rethink : Doc
        # Validate output directory
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'The parent of the GeoTiff file path is not a directory:'
        )
        # Automatically choose write function depending on input format
        if isinstance(x, PointCloud):
            return GeoTiffIO.write_pcloud(x, path, **kwargs)
        elif isinstance(x, tuple):
            return GeoTiffIO.write_grid_as_geotiff(x[0], x[1], path, **kwargs)
        else:
            raise TypeError(
                'GeoTiffIO does not support writing of given input type '
                f'({type(x)}).'
            )

    @staticmethod
    def write_pcloud(pcloud, path, **kwargs):
        """
        Write a point cloud as a GeoTiff file located at given path.

        :param pcloud: The point cloud to be written. It can be None, in which
            case the kwargs dictionary must contain an X element giving
            a matrix of coordinates and an F element giving the matrix of
            features.
        :param path: Path where the GeoTiff file must be written.
        :type path: str
        :param kwargs: The key-word arguments governing the GeoTiff
            specification and its writing.
        :type kwargs: dict
        :return: Nothing
        """
        # Validate point cloud
        if not isinstance(pcloud, PointCloud):
            X = kwargs.get('X', None)
            if X is None or not isinstance(X, np.ndarray):
                raise TypeError(
                    'Given object will not be written to a GeoTiff file '
                    'because it is not a PointCloud.'
                )
        # Write output GeoTiff
        start = time.perf_counter()
        GeoTiffIO.write_pcloud_as_geotiff(pcloud, path, **kwargs)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'GeoTiff file written to "{path}" in {end-start:.3f} seconds.'
        )

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def write_pcloud_as_geotiff(pcloud, path, **kwargs):
        """
        Assist the :meth:`geotiff_io.GeoTiffIO.write` method.

        :return: Nothing, but writes the GeoTiff file.
        """
        # Extract matrix of coordinates and features of interest
        X = kwargs.get('X', None)
        if X is None:
            X = pcloud.get_coordinates_matrix()
        fnames = kwargs.get('fnames', None)
        if fnames is None and pcloud is not None:
            fnames = pcloud.get_features_names()
        F = kwargs.get('F', None)
        if F is None:
            F = pcloud.get_features_matrix(fnames)
        # Validate extracted coordinates and features
        if X is None:
            raise ValueError(
                'GeoTiff will not be written because there is no matrix of '
                'coordinates available.'
            )
        if F is None:
            raise ValueError(
                'GeoTiff will not be written because there is no matrix of '
                'features available.'
            )
        # Extract GeoTiff specification
        driver = kwargs.get('driver', 'GTiff')
        crs = kwargs.get('crs', None)
        if crs is None and pcloud is not None:
            crs = pcloud.get_crs()
        xres = kwargs.get('xres', None)
        yres = kwargs.get('yres', None)
        # Validate GeoTiff specification
        GeoTiffIO.validate_geotiff_spec(crs, driver, xres, yres)
        # Write
        GeoTiffIO._write_pcloud_as_geotiff(
            X, F, path, driver, crs, xres, yres
        )

    @staticmethod
    def _write_pcloud_as_geotiff(X, F, path, driver, crs, xres, yres):
        """
        Assist the :meth:`geotiff_io.GeoTiffIO.write` method.

        :return: Nothing, but writes the GeoTiff file.
        """
        # Generate raster
        width, height, window, transform, xmin, xmax, ymin, ymax = \
            GeoTiffIO.generate_raster(X, xres, yres)
        # Bands
        count = 1 if len (F.shape) == 1 else F.shape[1]  # Num bands
        # Grid of features
        Xgrid = np.array(np.meshgrid(
            np.linspace(xmin, xmax, width),
            np.linspace(ymin, ymax, height)
        )).T
        Fgrid = GeoTiffIO.build_fgrid_from_pcloud(
            Xgrid, X[:, :2], F, width, height, xres, yres
        )
        # Write the GeoTiff
        GeoTiffIO._write_grid_as_geotiff(
            Fgrid, path, driver, crs, width, height,
            window, count, str(F.dtype), transform
        )

    @staticmethod
    def write_grid_as_geotiff(X, Fgrid, path, **kwargs):
        # TODO Rethink : Doc
        # Extract GeoTiff specification
        driver = kwargs.get('driver', 'GTiff')
        crs = kwargs.get('crs', None)
        xres = kwargs.get('xres', None)
        yres = kwargs.get('yres', None)
        # Validate GeoTiff specification
        GeoTiffIO.validate_geotiff_spec(crs, driver, xres, yres)
        # Prepare raster
        width, height, window, transform, xmin, xmax, ymin, ymax = \
            GeoTiffIO.generate_raster(X, xres, yres)
        count = 1 if len(Fgrid.shape) == 2 else Fgrid.shape[0]  # Bands
        dtype = str(Fgrid.dtype)
        # Validate raster is compatible with Fgrid
        if Fgrid.shape[-2] != height:
            raise ValueError(
                f'GeoTiffIO received a Fgrid with {Fgrid.shape[-2]} rows '
                f'but the raster width is {height}.'
            )
        if Fgrid.shape[-1] != width:
            raise ValueError(
                f'GeoTiffIO received a Fgrid with {Fgrid.shape[-1]} columns '
                f'but the raster width is {width}.'
            )
        # Write
        GeoTiffIO._write_grid_as_geotiff(
            Fgrid, path, driver, crs, width, height,
            window, count, dtype, transform
        )

    @staticmethod
    def _write_grid_as_geotiff(
        Fgrid, path, driver, crs, width, height,
        window, count, dtype, transform
    ):
        """
        Write the given grid of features to a GeoTiff file.

        :param Fgrid: The grid of features to be written. It has
            width x height features.
        :param path: The path where the GeoTiff file must be written.
        :param driver: The driver for the writer.
        :param crs: The coordinate reference system (CRS).
        :param width: The number of cells along the x-axis.
        :param height: The number of cells along the y-axis.
        :param window: The indexed window defining the boundary.
        :param count: The number of bands.
        :param dtype: The type of value (typically float).
        :param transform: The spatial transformation for the raster.
        :return: Nothing, but the GeoTiff file is written.
        """
        # Extract the indices
        indexes = [1] if count == 1 else [i+1 for i in range(count)]
        # Write the GeoTiff
        with rasterio.open(
            path, 'w', driver=driver, crs=crs, width=width, height=height,
            count=count, dtype=dtype, transform=transform
        ) as geo_tiff:
            geo_tiff.write(
                Fgrid,
                indexes=indexes,
                window=window
            )

    @staticmethod
    def build_fgrid_from_pcloud(Xgrid, X2D, F, width, height, xres, yres):
        """
        Build a grid of features from the given point cloud data.

        :param Xgrid: The grid of 2D coordinates.
        :param X2D: The 2D coordinates matrix representing the point cloud.
        :param F: The feature matrix representing the point cloud.
        :param width: The number of cells along the x-axis.
        :param height: The number of cells along the y-axis.
        :param xres: The step size along the x-axis.
        :param yres: The step size along the y-axis.
        :return: The grid of features corresponding to the grid of 2D
            coordinates and the given matrix of features.
        :rtype: :class:`np.ndarray`
        """
        # TODO Remove section ---
        np.savetxt(
            '/tmp/Xgrid.xyz',
            np.vstack(Xgrid)
        )
        # --- TODO Remove section
        # Prepare spatial queries
        kdt = KDT(X2D)
        radius = max(xres, yres)
        # Compute grid of features
        Fgrid = []
        for i in range(width):  # Iterate over rows
            Xi = Xgrid[i]
            I = KDT(Xi).query_ball_tree(kdt, radius)
            Fgrid.append(
                [np.mean(F[I[j]], axis=0) for j in range(height)]
            )
        return np.array(Fgrid).T

    @staticmethod
    def validate_geotiff_spec(crs, driver, xres, yres):
        """
        Validate whether the given GeoTiff specification is correct or not.
        
        :return: Nothing, but an Exception will be thrown if the specification
            is not valid.
        """
        if driver is None:
            raise ValueError(
                'GeoTiff cannot be written with None driver.'
            )
        if crs is None:
            raise ValueError(
                'GeoTiff cannot be written with None coordinate reference '
                'system.'
            )
        if xres is None:
            raise ValueError(
                'GeoTiff cannot be written with None horizontal (x) '
                'resolution.'
            )
        if yres is None:
            raise ValueError(
                'GeoTiff cannot be written with None vertical (y) '
                'resolution.'
            )

    @staticmethod
    def generate_raster(X, xres, yres):
        # TODO Rethink : Doc
        # Extract 2D bounding box
        X2D = X[:, :2]
        xmin, ymin = np.min(X2D, axis=0)
        xmax, ymax = np.max(X2D, axis=0)
        # Compute raster dimensions
        width = int((xmax-xmin) / xres)
        height = int((ymax-ymin) / yres)
        # Create transformation
        transform = rasterio.transform.from_origin(xmin, ymax, xres, yres)
        # Compute point-wise indices
        Icol = ((X[:, 0] - xmin) / xres).astype(int)  # Column indices
        Irow = ((ymax - X[:, 1]) / yres).astype(int)  # Row indices
        # Determine the indexed window
        window = ((min(Irow), max(Irow)), (min(Icol), max(Icol)))
        # Return
        return width, height, window, transform, xmin, xmax, ymin, ymax

