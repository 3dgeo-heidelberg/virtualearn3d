# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class SmoothFeatsMiner(Miner):
    r"""
    :author: Alberto M. Esmoris Pena

    Basic smooth features miner.
    See :class:`.Miner`.

    The smooth features miner considers each point in the point cloud
    :math:`\pmb{x_{i*}}` and finds either each knn or its spherical
    neighborhood :math:`\mathcal{N}`. Now, let :math:`j` index the points
    in the neighborhood. For then, a given feature :math:`f` can be
    smoothed by considering all the points in the neighborhood. In the most
    simple way, the smoothed feature :math:`\hat{f}` can be computed as a mean:

    .. math::

        \hat{f}_i = \dfrac{1}{\lvert\mathcal{N}\rvert}
            \sum_{j=1}^{\lvert\mathcal{N}\rvert}{f_j}

    Alternatively, the feature can be smoothed considering a weighted mean
    where the closest points with respect to :math:`\pmb{x_{i*}}` have a
    greater weight, such that:

    .. math::

        \hat{f}_i = \dfrac{1}{D}\sum_{j=1}^{\lvert\mathcal{N}\rvert}{d_j f_j}

    Where
    :math:`d^*=\max_{j} \; \left\{\lVert\pmb{x_{i*}} - \pmb{x_{j*}}\rVert : j = 1,\ldots,\lvert\mathcal{N}\rvert \right\}`,
    :math:`d_j = d^* - \lVert{\pmb{x_{i*}}-\pmb{x_{j*}}}\rVert + \omega`, and
    :math:`D = \sum_{j=1}^{\mathcal{N}}{d_j}`.

    Moreover, a Radial Basis Function (RBF) Gaussian kernel can be used to
    smooth the features in a given neighborhood such that:

    .. math::

        \hat{f}_i = \dfrac{1}{D} \sum_{j=1}^{\lvert\mathcal{N}\rvert}{
            \exp\left[
                - \dfrac{\lVert{\pmb{x_{i*}} - \pmb{x_{j*}}}\rVert^2}{\omega^2}
            \right]
        }

    Where
    :math:`D = \displaystyle\sum_{j=1}^{\lvert\mathcal{N}\rvert}{\exp\left[-\dfrac{\lVert\pmb{x_{i*}}-\pmb{x_{j*}}\rVert^2}{\omega^2}\right]}`
    .

    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a SmoothFeatsMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a SmoothFeatsMiner.
        """
        # Initialize
        kwargs = {
            'chunk_size': spec.get('chunk_size', None),
            'subchunk_size': spec.get('subchunk_size', None),
            'neighborhood': spec.get('neighborhood', None),
            'infnames': spec.get('infnames', None),
            'fnames': spec.get('fnames', None),
            'frenames': spec.get('frenames', None),
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of SmoothFeatsMiner.

        The neighborhood definition and feature names (fnames) are always
        assigned during initialization. The default neighborhood is a knn
        neighborhood with :math:`k=16`.

        :param kwargs: The attributes for the SmoothFeatsMiner that will also
            be passed to the parent.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the SmoothFeatsMiner
        self.chunk_size = kwargs.get('chunk_size', 100000)
        self.subchunk_size = kwargs.get('subchunk_size', 100)
        self.neighborhood = kwargs.get('neighborhood', {
            'type': 'knn',
            'k': 16
        })
        self.input_fnames = kwargs.get('input_fnames', None)
        self.fnames = kwargs.get(
            'fnames',
            ['mean', 'mean_weighted', 'gaussian_rbf']
        )
        self.frenames = kwargs.get('frenames', None)
        if self.frenames is None:
            neighborhood_type = self.neighborhood['type']
            neighborhood_type_low = neighborhood_type.lower()
            if neighborhood_type_low == 'knn':
                self.frenames = [
                    f'{fname}_k{self.neighborhood["k"]}'
                    for fname in self.fnames
                ]
            elif neighborhood_type_low == 'sphere':
                self.frenames = [
                    f'{fname}_r{self.neighborhood["radius"]}'
                    for fname in self.fnames
                ]
        # Validate attributes
        if self.input_fnames is None:
            raise MinerException(
                'SmoothFeatsMiner cannot be built without input features.'
            )

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine smooth features from the given point cloud.

        :param pcloud: The point cloud to be miend.
        :return: The point cloud extended with smooth features.
        :rtype: :class:`.PointCloud`
        """
        return pcloud  # TODO Rethink : Implement
