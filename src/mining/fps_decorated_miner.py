# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import Miner, MinerException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.utils.ptransf.fps_decorator_transformer import FPSDecoratorTransformer
from src.main.main_mine import MainMine
import time


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratedMiner(Miner):
    """
    :author: Alberto M. Esmoris Pena

    Decorator for data miners that makes the data mining process on an
    FPS-based representation of the point cloud.

    The FPS Decorated Miner (:class:`.FPSDecoratedMiner`)
    constructs a representation of the point cloud, then it runs the data
    mining process on this representation and, finally, it propagates the
    features back to the original point cloud.

    See :class:`.FPSDecoratorTransformer`.

    :ivar decorated_miner_spec: The specification of the decorated miner.
    :vartype decorated_miner_spec: dict
    :ivar decorated_miner: The decorated miner object.
    :vartype decorated_miner: :class:`.Miner`
    :ivar fps_decorator_spec: The specification of the FPS transformation
        defining the decorator.
    :vartype fps_decorator_spec: dict
    :ivar fps_decorator: The FPS decorator to be applied on input point clouds.
    :vartype fps_decorator: :class:`.FPSDecoratorTransformer`.
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a FPSDecoratedMiner
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a FPSDecoratedMiner.
        """
        # Initialize
        kwargs = {
            'decorated_miner': spec.get('decorated_miner', None),
            'fps_decorator': spec.get('fps_decorator', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type :class:`.FPSDecoratedMiner`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the FPSDecoratedMiner
        self.decorated_miner_spec = kwargs.get('decorated_miner', None)
        self.fps_decorator_spec = kwargs.get('fps_decorator', None)
        self.out_prefix = None
        # Validate decorated miner as an egg
        if self.decorated_miner_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedMiner did not receive any data mining specification.'
            )
            raise MinerException(
                'FPSDecoratedMiner did not receive any data mining specification.'
            )
        # Validate fps_decorator as an egg
        if self.fps_decorator_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedMiner did not receive any decorator specification.'
            )
            raise MinerException(
                'FPSDecoratedMiner did not receive any decorator specification.'
            )
        # Hatch validated miner egg
        miner_class = MainMine.extract_miner_class(self.decorated_miner_spec)
        self.decorated_miner = miner_class(
            **miner_class.extract_miner_args(self.decorated_miner_spec)
        )
        # Hatch validated fps_decorator egg
        self.fps_decorator = FPSDecoratorTransformer(**self.fps_decorator_spec)
        # Warn about not recommended number of encoding neighbors
        nen = self.fps_decorator_spec.get('num_encoding_neighbors', None)
        if nen is None or nen != 1:
            LOGGING.LOGGER.warning(
                f'FPSDecoratedMiner received {nen} encoding neighbors. '
                'Using a value different than one is not recommended as it '
                'could easily lead to unexpected behaviors.'
            )
        # TODO Pending : This logic is shared by fps_decorated_model
        # Maybe it can be abstracted to a common implementation.

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Decorate the main data mining logic to work on the representation.
        See :class:`.Miner` and :meth:`.Miner.mine`.
        """
        # Build representation from input point cloud
        start = time.perf_counter()
        rf_pcloud = self.fps_decorator.transform_pcloud(
            pcloud,
            fnames=getattr(self.decorated_miner, 'input_fnames', []),
            ignore_y=True,
            out_prefix=self.out_prefix
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'FPSDecoratedMiner computed a representation of '
            f'{pcloud.get_num_points()} points using '
            f'{rf_pcloud.get_num_points()} points '
            f'for data mining in {end-start:.3f} seconds.'
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Mine the point cloud
        start = time.perf_counter()
        F = Miner.get_feature_space_matrix(
            self.decorated_miner.mine(rf_pcloud),
            self.decorated_miner.frenames
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'{self.decorated_miner.__class__.__name__} '
            f'mined on representation in {end-start:.3f} seconds.'
        )
        # Delete representation
        rf_pcloud = None
        # Propagate mined features back to original space
        start = time.perf_counter()
        F = self.fps_decorator.propagate(F)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'FPSDecoratedMiner propagated '
            f'{F.shape[1] if len(F.shape) > 1 else 1} features in '
            f'{end-start:.3f} seconds.'
        )
        # Return point cloud extended with propagated features
        return pcloud.add_features(
            self.decorated_miner.frenames, F, ftypes=F.dtype
        )

    def get_decorated_fnames(self):
        """
        Get the feature names (``fnames``) from the decorated miner.
        :return: The feature names from the decorated miner.
        :rtype: list of str
        """
        if hasattr(self.decorated_miner, 'get_decorated_fnames'):
            return self.decorated_miner.get_decorated_fnames()
        return self.decorated_miner.fnames

    def get_decorated_frenames(self):
        """
        Get the feature renames (``frenames``) from the decorated miner.

        :return: The feature renames from the decorated miner, i.e., the
            names for the mined features.
        :rtype: list of str
        """
        if hasattr(self.decorated_miner, 'get_decorated_frenames'):
            return self.decorated_miner.get_decorated_frenames()
        return self.decorated_miner.frenames
