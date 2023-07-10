# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer, \
    FeatureTransformerException
from src.utils.dict_utils import DictUtils
from src.report.pca_projection_report import PCAProjectionReport
from src.plot.pca_variance_plot import PCAVariancePlot
import src.main.main_logger as LOGGING
from sklearn.decomposition import PCA
import laspy
import copy
import time


# ---   CLASS   --- #
# ----------------- #
class PCATransformer(FeatureTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming features by projecting them to a lower
    dimensionality space defined by the singular vectors of the centered
    matrix of features.

    :ivar out_dim: The number of features after the projection, i.e., the
        dimensionality of the output. It can be given as a float inside [0, 1]
        that represents how many variance must be preserved (1 preserves the
        100%, 0 nothing).
    :vartype out_dim: int or float
    :ivar whiten: True to multiply the singular vectors by the square root of
        the number of points and divide by the corresponding singular value.
        False otherwise.
    :vartype whiten: False
    :ivar random_seed: Optional attribute to specify a fixed random seed for
        the random computations of the model.
    :vartype random_seed: int
    :ivar frenames: The names for the output features (it must match the output
        dimensionality. If None, they will be determined automatically as
        PCA_{1}, ..., PCA_{out_dim}.
    :vartype frenames: list
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a PCATransformer.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a PCATransformer.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of PCATransformer
        kwargs['out_dim'] = spec.get('out_dim', None)
        kwargs['whiten'] = spec.get('whiten', None)
        kwargs['random_seed'] = spec.get('random_seed', None)
        kwargs['frenames'] = spec.get('frenames', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a PCATransformer

        :param kwargs: The attributes for the PCATransformer
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.out_dim = kwargs.get('out_dim', None)
        self.cached_out_dim = self.out_dim
        self.whiten = kwargs.get('whiten', False)
        self.random_seed = kwargs.get('random_seed', None)
        self.frenames = kwargs.get('frenames', None)

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the PCA transformer.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Transform
        in_dim = F.shape[1]
        start = time.perf_counter()
        pca = PCA(
            n_components=self.out_dim,
            whiten=self.whiten,
            random_state=self.random_seed,
            copy=True
        )
        F = pca.fit_transform(F)
        end = time.perf_counter()
        self.cached_out_dim = F.shape[1]
        # Validate output dimensionality
        if type(self.out_dim) is int and self.cached_out_dim != self.out_dim:
            raise FeatureTransformerException(
                f'The actual output dimensionality is {self.cached_out_dim} '
                f'but {self.out_dim} was requested.'
            )
        # Report explained variance
        self.report(
            PCAProjectionReport(
                self.get_names_of_transformed_features(),
                pca.explained_variance_ratio_,
                in_dim
            ),
            out_prefix=out_prefix
        )
        # Plot explained variance
        if self.plot_path is not None:
            PCAVariancePlot(
                pca.explained_variance_ratio_,
                path=self.plot_path
            ).plot(out_prefix=out_prefix)
        # Log transformation
        LOGGING.LOGGER.info(
            'PCATransformer transformed {n1} features into {n2} features in '
            '{texec:.3f} seconds.'.format(
                n1=in_dim,
                n2=self.cached_out_dim,
                texec=end-start
            )
        )
        # Return
        return F

    def get_names_of_transformed_features(self, **kwargs):
        """
        Obtain the names that correspond to the transformed features.

        :return: The list of strings representing the names of the transformed
            features.
        :rtype: list
        """
        new_fnames = self.frenames
        if new_fnames is None:  # Default PCA_{i} for i = 1, ..., out_dim
            new_fnames = [f'PCA_{i+1}' for i in range(self.cached_out_dim)]
        return new_fnames

    def build_new_las_header(self, pcloud):
        """
        See
        :meth:`feature_transformer.FeatureTransformer.build_new_las_header`.
        """
        # Obtain header
        header = copy.deepcopy(pcloud.las.header)
        # Remove old features
        header.remove_extra_dims(self.fnames)
        # Add PCA features
        extra_bytes = [
            laspy.ExtraBytesParams(name=frename, type='f')
            for frename in self.get_names_of_transformed_features()
        ]
        header.add_extra_dims(extra_bytes)
        # Return
        return header
