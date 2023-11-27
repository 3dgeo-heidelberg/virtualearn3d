# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import time


class TakeClosestMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena

    Take closest miner.
    See :class:`.Miner`.

    The take closest miner considers a pool of point clouds and for each point
    in the input point cloud takes the requested features from the closest
    neighbor in the entire pool. It can be useful, for example, to have
    a set of mined point clouds and take just some points for training that
    have been manually labeled in the non mined point clouds (e.g., uncertainty
    point clouds, see :class:`.ClassificationUncertaintyEvaluator`).
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a TakeClosestMiner from
        a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a TakeClosestMiner.
        """
        # Initialize
        kwargs = {
            'fnames': spec.get('fnames', None),
            'pcloud_pool': spec.get('pcloud_pool', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of TakeClosestMiner.

        :param kwargs: The attributes for the TakeClosestMiner that will also
            be passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the TakeClosestMiner
        self.fnames = kwargs.get('fnames', None)
        self.pcloud_pool = kwargs.get('pcloud_pool', None)
        if isinstance(self.pcloud_pool, str):
            self.pcloud_pool = [self.pcloud_pool]
        # Validate attributes
        if self.fnames is None:
            raise MinerException(
                'TakeClosestMiner cannot be computed without feature names.'
            )
        if self.pcloud_pool is None:
            raise MinerException(
                'TakeClosestMiner cannot be computed without a pool of '
                'point clouds.'
            )
