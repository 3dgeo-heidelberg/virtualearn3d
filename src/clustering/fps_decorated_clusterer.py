# ---   IMPORTS   --- #
# ------------------- #
from src.clustering.clusterer import Clusterer, ClusteringException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.utils.ptransf.fps_decorator_transformer import FPSDecoratorTransformer
from src.main.main_clustering import MainClustering
import time


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratedClusterer(Clusterer):
    """
    :author: Alberto M. Esmoris Pena

    Decorator for clusterers that makes the clustering process on an FPS-based
    representation of the point cloud.

    The FPS Decorated Clusterer (:class:`.FPSDecoratedClusterer`)
    constructs a representation of the point cloud, then it runs the clustering
    process on this representation and, finally, it propagates the clusters
    back to the original point cloud.

    See :class:`.FPSDecoratorTransformer`.

    :ivar decorated_clusterer_spec: The specification of the decorated
        clusterer.
    :vartype decorated_clusterer_spec: dict
    :ivar fps_decorator_spec: The specification of the FPS transformation
        defining the decorator.
    :vartype fps_decorator_spec: dict
    :ivar fps_decorator: The FPS decorator to be applied on input point clouds.
    :vartype fps_decorator: :class:`.FPSDecoratorTransformer`
    """
    # ---  SPECIFICAITON ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_clustering_args(spec):
        """
        Extract the arguments to initialize/instantiate a FPSDecoratedClusterer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            FPSDecoratedClusterer.
        """
        # Extract arguments from parent
        kwargs = Clusterer.extract_clustering_args(spec)
        # Update arguments with those from FPSDecoratedClusterer
        kwargs['decorated_clusterer'] = spec.get('decorated_clusterer', None)
        kwargs['fps_decorator'] = spec.get('fps_decorator', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type
        :class:`.FPSDecoratedClusterer`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of FPSDecoratedClusterer
        self.decorated_clusterer_spec = kwargs.get('decorated_clusterer', None)
        self.fps_decorator_spec = kwargs.get('fps_decorator', None)
        # Validate decorated clusterer as an egg
        if self.decorated_clusterer_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedClusterer did not receive any clustering '
                'specification.'
            )
            raise ClusteringException(
                'FPSDecoratedClusterer did not receive any clustering '
                'specification.'
            )
        # Validate fps_decorator as an egg
        if self.fps_decorator_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedClusterer did not receive any decorator '
                'specification.'
            )
            raise ClusteringException(
                'FPSDecoratedClusterer did not receive any decorator '
                'specification.'
            )
        # Hatch validated clusterer egg
        clusterer_class = MainClustering.extract_clusterer_class(
            self.decorated_clusterer_spec
        )
        self.decorated_clusterer = clusterer_class(
            **clusterer_class.extract_clustering_args(
                self.decorated_clusterer_spec
            )
        )
        # Hatch validated fps_decorator egg
        self.fps_decorator = FPSDecoratorTransformer(**self.fps_decorator_spec)
        # Warn about not recommended number of encoding neighbors
        nen = self.fps_decorator_spec.get('num_encoding_neighbors', None)
        if nen is None or nen != 1:
            LOGGING.LOGGER.warning(
                f'FPSDecoratedClusterer received {nen} encoding neighbors. '
                'Using a value different than one is not recommended as it '
                'could easily lead to unexpected behaviors.'
            )
        # TODO Pending : This logic is shared by fps_decorated_miner and
        # fps_decorated_model. Perhaps abstract it to common implementation.

    # ---   CLUSTERING METHODS   --- #
    # ------------------------------ #
    def fit(self, pcloud):
        """
        Fit a clustering model to a given input point cloud.

        In doing so, the clustering model is fit to an FPS representation of
        the input point cloud.

        See :class:`.Clusterer` and :meth:`.Clusterer.fit`.
        """
        self.decorated_clusterer.fit(self.transform_pcloud(pcloud))
        return self

    def cluster(self, pcloud):
        """
        Clustering from a given input point cloud.

        In doing so, the clustering model is computed on an FPS representation
        of the input point cloud.

        See :class:`.Clusterer` and :meth:`.Clusterer.cluster`.
        """
        fps_pcloud = self.transform_pcloud(pcloud)
        fps_pcloud = self.decorated_clusterer.cluster(fps_pcloud)
        return self.propagate(fps_pcloud, pcloud)

    def post_process(self, pcloud):
        """
        Run the post-processing pipeline on the given input point cloud.

        In doing so, the post-processing pipeline is computed on an FPS
        representation of the input point cloud.

        See :class:`.Clusterer` and :meth:`.Clusterer.post_process`.
        """
        fps_pcloud = self.transform_pcloud(pcloud)
        pcloud = self.decorated_clusterer.post_process(pcloud)
        return self.propagate(fps_pcloud, pcloud)

    def fit_cluster_and_post_process(self, pcloud, out_prefix=None):
        """
        Compute the fitting, clustering, and post-processing as a whole.

        In doing so, the FPS representation of the input point cloud is
        computed once and used to fit, cluster, and post-process with the
        propagations applied after all the previous methods have been called.

        See :class:`.Clusterer` and
        :meth:`.Clusterer.fit_cluster_and_post_process`.
        """
        fps_pcloud = self.transform_pcloud(pcloud)
        fps_pcloud = self.decorated_clusterer.fit_cluster_and_post_process(
            fps_pcloud, out_prefix=out_prefix
        )
        return self.propagate(fps_pcloud, pcloud)

    # ---  FPS DECORATION METHODS  --- #
    # -------------------------------- #
    def transform_pcloud(self, pcloud):
        """
        Transform the given input point cloud to its FPS representation.

        :param pcloud: The input point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :return: The FPS representation of the input point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Determine default feature names
        default_fnames = []
        if hasattr(self.decorated_clusterer, "precluster_name"):
            default_fnames.append(self.decorated_clusterer.precluster_name)
        # Build representation from input point cloud
        start = time.perf_counter()
        rf_pcloud = self.fps_decorator.transform_pcloud(
            pcloud,
            fnames=getattr(
                self.decorated_clusterer, 'input_fnames', default_fnames
            ),
            ignore_y=True,
            out_prefix=self.out_prefix
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'FPSDecoratedClusterer computed a representation of '
            f'{pcloud.get_num_points()} points using '
            f'{rf_pcloud.get_num_points()} points '
            f'for data mining in {end-start:.3f} seconds.'
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Return transformed point cloud
        return rf_pcloud

    def propagate(self, fps_pcloud, pcloud):
        """
        Propagate the cluster labels from the FPS version of the point cloud
        back to the original one.

        :param fps_pcloud: The FPS point cloud whose cluster labels must be
            propagated to the original point cloud.
        :type fps_pcloud: :class:`.PointCloud`
        :param pcloud: The original point cloud that will receive the cluster
            labels propagated from the FPS point cloud.
        :type pcloud: :class:`.PointCloud`
        :return: The original point cloud updated with the cluster labels
            from the FPS point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Propagate cluster labels back to original space
        start = time.perf_counter()
        c = fps_pcloud.get_features_matrix([
            self.decorated_clusterer.cluster_name
        ]).flatten()
        R = len(c)
        c = self.fps_decorator.propagate(c)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'FPSDecoratedClusterer propagated {R} cluster labels to '
            f'{len(c)} points in {end-start:.3f} seconds.'
        )
        # Return point cloud extended with propagated cluster labels
        self.cluster_name = self.decorated_clusterer.cluster_name
        return self.add_cluster_labels_to_point_cloud(pcloud, c)
