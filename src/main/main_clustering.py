# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.main.main_mine import MainMine
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
from src.inout.io_utils import IOUtils
from src.clustering.dbscan_clusterer import DBScanClusterer
import os
import time


# ---   CLASS   --- #
# ----------------- #
class MainClustering:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for clustering tasks.
    """
    # ---  MAIN METHOD  --- #
    # --------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for clustering tasks.

        :param spec: Key-word specification.
        """
        LOGGING.LOGGER.info('Starting clustering ...')
        start = time.perf_counter()
        clusterer_class = MainClustering.extract_clusterer_class(spec)
        clusterer = clusterer_class(
            **clusterer_class.extract_clusterer_args(spec)
        )
        pcloud = PointCloudFactoryFacade.make_from_file(
            MainClustering.extract_input_path(spec)
        )
        clusterer.fit(pcloud)
        pcloud = clusterer.cluster(pcloud)
        clusterer.post_process(pcloud)
        outpath = MainClustering.extract_output_path(spec)
        PointCloudIO.write(pcloud, outpath)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Clustering computed in {end-start:.3f} seconds. '
            f'Output point cloud written to "{os.path.abspath(outpath)}".'
        )

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_input_path(
        spec,
    ):
        """
        See :class:`.MainMine` and
        :func:`main_mine.MainMine.extract_input_path`.
        """
        return MainMine.extract_input_path(
            spec,
            none_path_msg='Clustering a point cloud requires an input point '
                          'cloud. None was given.',
            invalid_path_msg='Cannot find the input file for clustering.\n'
                             'Given path: {path}'
        )

    @staticmethod
    def extract_output_path(spec):
        """
        Extract the output path from the key-word specification.

        :param spec: The key-word specification.
        :return: Output path as string.
        :rtype: str
        """
        # TODO Rethink : Abstract to common logic together with MainMine.extract_output_path
        path = spec.get('out_pcloud', None)
        if path is None:
            raise ValueError(
                "Clustering a point cloud requires an output path to store the "
                "results. None was given."
            )
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the clustering point cloud:'
        )
        return path

    @staticmethod
    def extract_clusterer_class(spec):
        """
        Extract the clusterer's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a clusterer.
        :rtype: :class:`.Clusterer`
        """
        clusterer = spec.get('clustering', None)
        if clusterer is None:
            raise ValueError(
                "Clustering a point cloud requires a clusterer. "
                "None was specified."
            )
        # Check clusterer class
        clusterer_low = clusterer.lower()
        if clusterer_low == 'dbscan':
            return DBScanClusterer
        elif clusterer_low == 'fpsdecorated':
            from src.clustering.fps_decorated_clusterer \
                import FPSDecoratedClusterer
            return FPSDecoratedClusterer
        # An unknown clusterer was specified
        raise ValueError(f'There is no known clusterer "{clusterer}"')
