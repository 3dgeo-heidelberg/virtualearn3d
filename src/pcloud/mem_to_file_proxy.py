# ---   IMPORTS   --- #
# ------------------- #
from src.utils.sys_utils import SysUtils
import joblib
import tempfile
import laspy


# ---   CLASS   --- #
# ----------------- #
class MemToFileProxy:
    """
    :author: Alberto M. Esmoris Pena

    Class representing a proxy between point clouds loaded in memory and a
    binary file in persistent storage representing them. The proxy must support
    the operation of loading a point cloud from its binary representation,
    the operation of saving a point cloud from its binary representation, and
    the logic to check when the data of a given point cloud must be dumped
    to save memory sources.

    :ivar mem_check_threshold: Decimal number inside [0, 1] that determines
        the cut value (threshold) of occupied memory that is acceptable
        before dumping the point cloud's data. More concretely, when the
        memory required by the point cloud's data divided by the system's
        memory is greater than this threshold, dumping the point cloud data
        will be recommended.
    :vartype mem_check_threshold: float
    :ivar proxy_file: The binary proxy file where the point cloud's data is
        stored.
    :vartype proxy_file: None or file object
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, mem_check_threshold=0.00034):
        # General attributes
        self.mem_check_threshold = mem_check_threshold
        # Cache-like attributes
        self.proxy_file = None

    # ---  MEM TO FILE METHODS  --- #
    # ----------------------------- #
    def dump(self, pcloud):
        """
        Dump the given point cloud's data to the binary proxy file.

        :param pcloud: The point cloud which data must be dumped.
        :type pcloud: :class:`.PointCloud`
        """
        # Create binary file
        self.proxy_file = tempfile.TemporaryFile("a+b")
        # Dump memory to binary proxy file
        pcloud.las.write(self.proxy_file)  # TODO Rethink : Alternative
        #joblib.dump(pcloud.las, self.proxy_file)  # TODO Rethink : Baseline
        self.proxy_file.flush()
        # Remove data from point cloud
        pcloud.las = None

    def load(self, pcloud):
        """
        Load the data from the binary proxy file into the given point cloud.

        :param pcloud: Point cloud for which the data in the proxy file must be
            loaded.
        :type pcloud: :class:`.PointCloud`
        """
        # Read from binary file
        self.proxy_file.seek(0)  # Jump to the beginning
        pcloud.las = laspy.read(self.proxy_file)  # TODO Rethink : Alternative
        #pcloud.las = joblib.load(self.proxy_file)  # TODO Rethink : Baseline
        # Discard binary proxy file
        self.proxy_file.close()
        self.proxy_file = None

    def is_dump_recommended(self, pcloud):
        """
        Check whether dumping the given point cloud is recommended (True) or
        not (False). A point cloud must be dumped when the memory it requires
        is greater than self.mem_check_threshold considering the system's
        memory.

        See :class:`.SysUtils` and :meth:`SysUtils.get_sys_mem`.

        :param pcloud: The point cloud to be checked.
        :type pcloud: :class:`.PointCloud`
        :return: True if the given point cloud should be dumped, False
            otherwise.
        :rtype: bool
        """
        num_points = len(pcloud.las.X)
        num_attributes = len(pcloud.las.point_format.dimensions)
        mem_pcloud = num_points * num_attributes * 8  # Assume 8 bytes per attr
        mem_ratio = mem_pcloud / SysUtils.get_sys_mem()
        return mem_ratio >= self.mem_check_threshold

    def is_dumped(self):
        """
        Check whether the proxy is representing a dumped point cloud or not.

        :return: True if the proxy holds a dumped point cloud, false otherwise.
        :rtype: bool
        """
        return self.proxy_file is not None
