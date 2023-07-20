# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
from src.inout.io_utils import IOUtils
from src.mining.geom_feats_miner import GeomFeatsMiner
from src.mining.covar_feats_miner import CovarFeatsMiner
import os
import time


# ---   CLASS   --- #
# ----------------- #
class MainMine:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for data mining tasks.
    """
    # ---  MAIN METHOD  --- #
    # --------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for data mining tasks

        :param spec: Key-word specification
        """
        LOGGING.LOGGER.info('Starting data mining ...')
        start = time.perf_counter()
        pcloud = PointCloudFactoryFacade.make_from_file(
            MainMine.extract_input_path(spec)
        )
        miner_class = MainMine.extract_miner_class(spec)
        miner = miner_class(**miner_class.extract_miner_args(spec))
        pcloud = miner.mine(pcloud)
        outpath = MainMine.extract_output_path(spec)
        PointCloudIO.write(pcloud, outpath)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Data mining computed in {end-start:.3f} seconds. '
            f'Output point cloud written to "{os.path.abspath(outpath)}".'
        )

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_input_path(
        spec,
        none_path_msg='Mining a point cloud requires an input point cloud. '\
            'None was given.',
        invalid_path_msg='Cannot find the input file for datamining.\n'\
            'Given path: {path}'
    ):
        """
        Extract the input path from the key-word specification.

        :param spec: The key-word specification.
        :type spec: dict
        :param none_path_msg: The string representing the message to be logged
            when no input point cloud is given.
        :type none_path_msg: str
        :param invalid_path_msg: The string representing the message to be
            logged when the given input path is not valid. It supports a
            "{path}" format field that will be replaced by the invalid path
            in the message.
        :return: Input path as string.
        :rtype: str
        """
        path = spec.get('in_pcloud', None)
        if path is None:
            raise ValueError(none_path_msg)
        IOUtils.validate_path_to_file(
            path,
            invalid_path_msg.format(path=path),
            accept_url=True
        )
        return path

    @staticmethod
    def extract_output_path(spec):
        """
        Extract the output path from the key-word specification.

        :param spec: The key-word specification.
        :return: Output path as string.
        :rtype: str
        """
        path = spec.get('out_pcloud', None)
        if path is None:
            raise ValueError(
                "Mining a point cloud requires an output path to store the "
                "results. None was given."
            )
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the mined features:'
        )
        return path

    @staticmethod
    def extract_miner_class(spec):
        """
        Extract the miner's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a miner.
        :rtype: :class:`.Miner`
        """
        miner = spec.get('miner', None)
        if miner is None:
            raise ValueError(
                "Mining a point cloud requires a miner. None was specified."
            )
        # Check miner class
        miner_low = miner.lower()
        if miner_low == 'geometricfeatures':
            return GeomFeatsMiner
        if miner_low == "covariancefeatures":
            return CovarFeatsMiner
        # An unknown miner was specified
        raise ValueError(f'There is no known miner "{miner}"')
