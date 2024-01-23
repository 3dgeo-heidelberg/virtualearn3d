# ---   IMPORTS  --- #
# ------------------ #
from src.utils.ftransf.feature_transformer import FeatureTransformer, \
    FeatureTransformerException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class ExplicitSelector(FeatureTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming features by discarding or preserving exactly those
    given as input.

    :ivar fnames: The names of the features that must be either discarded
        or preserved.
    :vartype fnames: list of str
    :ivar preserve: The flag governing whether to preserve the given features
        (True, default) or not (False).
    :vartype preserve: bool
    """
    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate an ExplicitSelector.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an ExplicitSelector.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of ExplicitSelector
        kwargs['preserve'] = spec.get('preserve', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ExplicitSelector.

        :param kwargs: The attributes for the ExplicitSelector.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.preserve = kwargs.get('preserve', True)
        # Validate attributes
        if self.fnames is None:
            raise FeatureTransformerException(
                'ExplicitSelector cannot be initialized without explicit '
                'feature names (fnames).'
            )

    # ---   FEATURE TRANSFORM METHODS  --- #
    # ------------------------------------ #
    def transform(
        self, F, y=None, fnames=None, out_prefix=None, F_fnames=None
    ):
        """
        The fundamental feature transform logic defining the explicit
        selector.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.

        Note that, to the contrary of many other feature transformers, the
        logic in this transform method is not adequate to be called from the
        corresponding transform_pcloud (as it has been overriden to efficiently
        deal with point clouds as input).

        :param F_fnames: The names of the features (columns) for the input
            feature space matrix F.
        :type F_fnames: list of str
        """
        start = time.perf_counter()
        old_num_features = F.shape[1]
        # Validate
        if F_fnames is None:
            raise FeatureTransformerException(
                'ExplicitSelector transform on a matrix of features without '
                'given F_fnames is not supported.'
            )
        # Get fnames
        if fnames is None:
            fnames = self.fnames
        if fnames is None:
            raise FeatureTransformerException(
                'ExplicitSelector cannot apply any transformation without '
                'explicit feature names (fnames).'
            )
        # Determine indices of preserved features
        if self.preserve:  # Preserve explicit fnames
            feat_ids = [
                i for i in range(F.shape[1]) if F_fnames[i] in fnames
            ]
        else:  # Discard explicit fnames
            feat_ids = [
                i for i in range(F.shape[1]) if F_fnames[i] not in fnames
            ]
        # Update the matrix of features
        F = F[:, feat_ids]
        end = time.perf_counter()
        # Log transformation
        new_num_features = F.shape[1]
        LOGGING.LOGGER.info(
            f'ExplicitSelector transformed {old_num_features} into '
            f'{new_num_features} in {end-start:.3f} seconds'
        )
        # Return
        return F

    def transform_pcloud(self, pcloud, out_prefix=None, fnames=None):
        """
        Apply the explicit selector to a point cloud, overwriting the parent's
        logic to send the names of the features (columns) of the matrix F.

        See :meth:`feature_transformer.FeatureTransformer.transform_pcloud`.
        """
        start = time.perf_counter()
        pcloud_fnames = pcloud.get_features_names()
        old_num_features = len(pcloud_fnames)
        # Check feature names
        fnames = self.safely_handle_fnames(fnames=fnames)
        # Filter feature names
        if self.preserve:  # Preserve explicit fnames
            fnames_to_remove = [
                fname for fname in pcloud_fnames if fname not in fnames
            ]
        else:  # Discard explicit fnames
            fnames_to_remove = fnames
        # Update point cloud's features
        pcloud.remove_features(fnames_to_remove)
        # Log transformation
        new_num_features = len(pcloud.get_features_names())
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ExplicitSelector removed {old_num_features-new_num_features} '
            f'features from the point cloud (now with {new_num_features} '
            f'features) in {end-start:.3f} seconds.'
        )
        # Return updated point cloud
        return pcloud

