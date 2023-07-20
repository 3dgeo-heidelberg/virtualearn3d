# ---   IMPORTS   --- #
# ------------------- #
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud import PointCloud
import src.main.main_logger as LOGGING
from urllib import request as urlreq
import laspy
import os
import io


# ---   CLASS   --- #
# ----------------- #
class PointCloudIO:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for input/output operations related
    to point clouds.
    """

    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read(path):
        """
        Read a LAS/LAZ point cloud.

        :param path: Path pointing to a LAS/LAZ point cloud.
        :return: The read point cloud.
        :rtype: :class:`.PointCloud`
        """
        if IOUtils.is_url(path):
            return PointCloudIO.read_url(path)
        return PointCloudIO.read_path(path)

    @staticmethod
    def read_path(path):
        """
        Read a LAS/LAZ point cloud file.

        :param path: Path pointing to a LAS/LAZ point cloud file.
        :type path: str
        :return: The read point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find point cloud file at given input path:'
        )
        # Read and return point cloud
        return PointCloud(laspy.read(path))

    @staticmethod
    def read_url(url, chunk_size=64*1024**2):
        """
        Read a LAS/LAZ point cloud file from a given URL.

        :param url: The URL corresponding to a LAS/LAZ point cloud.
        :type url: str
        :param chunk_size: The chunk size for each reading operation on the
            input stream (in Bytes, def. 64 MiB).
        :type chunk_size: float
        :return: :class:`.PointCloud`
        """
        # Validate input as URL
        IOUtils.validate_url_to_file(
            url,
            'Cannot find point cloud at given input URL:'
        )
        # Read point cloud
        open_url = urlreq.urlopen(url)  # Open input stream ---
        max_bytes = int(open_url.info().get('content-length'))
        log_ratio_th, log_ratio_step = 0.0, 0.1
        pcloud_bytes = b''  # Initialize pcloud as a buffer of Bytes
        read_bytes = open_url.read(chunk_size)  # Read the first chunk
        while read_bytes != b'':  # While something has been read
            pcloud_bytes += read_bytes  # Merge read with pcloud buffer
            read_bytes = open_url.read(chunk_size)  # Next reading
            if max_bytes is not None:  # If possible, log completed percentage
                compl_ratio = len(pcloud_bytes) / max_bytes
                if compl_ratio > log_ratio_th:
                    LOGGING.LOGGER.info(
                        f'Downloaded {100*compl_ratio:.2f}% from URL "{url}"'
                    )
                    log_ratio_th += log_ratio_step
        open_url.close()  # --- Close input stream
        # Return built point cloud
        try:
            pcloud = PointCloud(laspy.read(io.BytesIO(pcloud_bytes)))
            LOGGING.LOGGER.info(
                f'Downloaded {len(pcloud_bytes)/1024**2} MiB from URL: "{url}"'
            )
            return pcloud
        except Exception as ex:
            LOGGING.LOGGER.error(f'Failed to load data from URL: "{url}"')
            raise ex

    # ---  WRITE / STORE  --- #
    # ----------------------- #
    @staticmethod
    def write(pcloud, path):
        """
        Write a LAS/LAZ point cloud file.

        :param pcloud: The point cloud to be written.
        :param path: Path where the LAS/LAZ file must be written.
        :return: Nothing
        """
        # Validate output directory
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'The parent of the output point cloud path is not a directory:'
        )
        # Write output point cloud
        pcloud.las.write(path)
