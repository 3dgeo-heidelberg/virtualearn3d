# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
from src.utils.dict_utils import DictUtils
import numpy as np
import joblib


# ---   CLASS   --- #
# ----------------- #
class HSVFromRGBMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena

    Mine Hue, Saturation and Value (HSV) components representing color from
    available Red, Green, Blue (RGB) components.
    See :class:`.Miner`.

    :ivar chunk_size: How many points per chunk must be considered for
        parallel executions.
    :vartype chunk_size: int
    :ivar nthreads: The number of threads to be used for the parallel
        computation of the geometric features. Note using -1 (default value)
        implies using as many threads as available cores.
    :vartype nthreads: int
    :ivar frenames: Optional attribute to specify how to rename the features
        representing the HSV components. The first element corresponds to
        Hue, the second to Saturation, and the third one to Value.
    :vartype frenames: list
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # ⨪-------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a HSVFromRGBMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a HSVFromRGBMiner.
        """
        # Initialize
        kwargs = {
            'chunk_size': spec.get('chunk_size', None),
            'hue_unit': spec.get('hue_unit', None),
            'frenames': spec.get('frenames', None),
            'nthreads': spec.get('nthreads', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of HSVFromRGBMiner.

        :param kwargs: The attributes for the HSVFromRGBMiner that will also be
            passed to the parent.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the HSVFromRGBMiner
        self.chunk_size = kwargs.get('chunk_size', 1000000)
        self.hue_unit = kwargs.get('hue_unit', 'radians')
        self.frenames = kwargs.get('frenames', None)
        self.nthreads = kwargs.get('nthreads', -1)
        # Optional attribute to rename the computed features
        if self.frenames is None:
            self.frenames = ['HSV_H', 'HSV_S', 'HSV_V']
        # TODO Rethink : Discard parallel execution (i.e., nthreads and chunk_size) ?

    # ---  MINER METHODS  --- #
    # ----------------------- #
    def mine(self, pcloud):
        """
        Mine geometric features from the given point cloud.

        :param pcloud: The point cloud to be miend
        :type pcloud: :class:`.PointCloud`
        :return: The point cloud extended with HSV color components as
            features.
        :rtype: :class:`.PointCloud`
        """
        # Check RGB is available
        if not pcloud.has_given_features(['red', 'green', 'blue']):
            raise MinerException(
                'HSVFromRGBMiner cannot extract HSV as features because there '
                'are no RGB components available in the point cloud.'
            )
        # Extract RGB components
        RGB = pcloud.get_features_matrix(['red', 'green', 'blue'])
        R, G, B = RGB[:, 0], RGB[:, 1], RGB[:, 2]
        RGB = None
        # Transform RGB to HSV
        H, S, V = HSVFromRGBMiner.RGB_to_HSV(
            R, G, B,
            nthreads=self.nthreads,
            chunk_size=self.chunk_size,
            hue_unit=self.hue_unit
        )
        HSV = np.hstack([H.reshape(-1, 1), S.reshape(-1, 1), V.reshape(-1, 1)])
        # Return point cloud extended with HSV as features
        return pcloud.add_features(self.frenames, HSV)

    # ---   HSV METHODS   --- #
    # ----------------------- #
    @staticmethod
    def RGB_to_HSV(R, G, B, nthreads=1, chunk_size=0, hue_unit='radians'):
        r"""
        Transform the received RGB components in :math:`[0, 1]` to HSV. If
        RGB components are given in :math:`[0, 255]` they will be automatically
        mapped to :math:`[0, 1]`. Also, if RGB components are given in
        :math:`[0, 65535]` they will be automatically mapped to :math:`[0, 1]`.

        :param R: The red component for each point.
        :param G: The green component for each point.
        :param B: The blue component for each point.
        :param nthreads: The number of threads to use for parallel execution
            (-1 means as many threads as available cores).
        :param chunk_size: The number of points per chunk for parallel
            execution (0 means no chunks at all, hence all tasks will be run
            in a single thread).
        :return: A tuple of three arrays representing Hue (H), Saturation (S)
            and Value (V).
        :rtype: tuple of :class:`np.ndarray`
        """
        # Normalize to [0, 1] when necessary
        if any(np.any(x > 1) for x in [R, G, B]):
            if not (
                any(np.any(x > 255) for x in [R, G, B]) or
                any(np.any(x < 0) for x in [R, G, B])
            ):
                R, G, B = R/255, G/255, B/255
            elif not(
                any(np.any(x > 65535) for x in [R, G, B]) or
                any(np.any(x < 0) for x in [R, G, B])
            ):
                R, G, B = R/65535, G/65535, B/65535
            else:
                raise MinerException(
                    'HSVFromRGBMiner.RGB_to_HSV failed because RGB components '
                    'were given in an unexpected format.\n'
                    f'(min, max) for R is ({np.min(R)}, {np.max(R)})\n'
                    f'(min, max) for G is ({np.min(G)}, {np.max(G)})\n'
                    f'(min, max) for B is ({np.min(B)}, {np.max(B)})\n'
                )
        # Transform (assuming components are in [0, 1] here) : Prepare
        RGB = np.hstack([R.reshape(-1, 1), G.reshape(-1, 1), B.reshape(-1, 1)])
        RGBmin = np.min(RGB, axis=1)
        RGBmax = np.max(RGB, axis=1)
        RGBrange = RGBmax-RGBmin
        # Transform : Hue
        H = np.zeros_like(R)
        RGBrange_nonzero_mask = RGBrange != 0
        mask = (RGBmax == R) * RGBrange_nonzero_mask
        H[mask] = 60 * np.mod((G[mask]-B[mask])/RGBrange[mask], 6)
        mask = (RGBmax == G) * RGBrange_nonzero_mask
        H[mask] = 60 * ((B[mask]-R[mask])/RGBrange[mask] + 2)
        mask = (RGBmax == B) * RGBrange_nonzero_mask
        H[mask] = 60 * ((R[mask]-G[mask])/RGBrange[mask] + 4)
        hue_unit_low = hue_unit.lower()
        if hue_unit_low == 'radians':
            H = H * np.pi / 180
        elif hue_unit_low != 'degrees':
            raise MinerException(
                'HSVFromRGBMiner.RGB_to_HSV received an unexpected hue unit: '
                f'"{hue_unit}"'
            )
        # Transform : Saturation
        S = np.zeros_like(R)
        mask = RGBmax > 0
        S[mask] = RGBrange[mask] / RGBmax[mask]
        # Transform : Value
        V = RGBmax
        # Return
        return H, S, V

