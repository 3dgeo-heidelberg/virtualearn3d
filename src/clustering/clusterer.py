# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import src.main.main_logger as LOGGING
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
import time


# ---  EXCEPTIONS  --- #
# -------------------- #
class ClusteringException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to clustering components
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Clusterer:
    """
    :author: Alberto M. Esmoris Pena.

    Interface governing any clustering component.

    :ivar cluster_name: The name for the computed clusters. It will be used
        to reference the cluster column in the output point cloud.
    :vartype cluster_name: str
    :ivar post_clustering:
    :vartype post_clustering: None or list of callable
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_clustering_args(spec):
        """
        Extract the arguments to initialize/instantiate a Clusterer from a
        key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a Clusterer.
        """
        # Initialize
        kwargs = {
            'cluster_name': spec.get('cluster_name', None),
            'post_clustering': spec.get('post_clustering', None),
            'out_prefix': spec.get('out_prefix', None)
        }
        #  Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a Clusterer.

        :param kwargs: The key-word arguments for the initialization of any
            Clusterer. It must contain the name of the cluster to be computed.
        """
        # Assign member attributes
        self.cluster_name = kwargs.get('cluster_name', 'CLUSTER')
        self.post_clustering = kwargs.get('post_clustering', None)
        self.out_prefix = kwargs.get('out_prefix', None)
        # Build post-processors
        if self.post_clustering is not None:
            from src.clustering.postproc.clustering_post_processor import \
                ClusteringPostProcessor
            self.post_clustering = [
                ClusteringPostProcessor.build_post_processor(spec)
                for spec in self.post_clustering
            ]

    # ---  CLUSTERING METHODS  --- #
    # ---------------------------- #
    def fit(self, pcloud):
        """
        Fit a clustering model to a given input point cloud.

        :param pcloud: The input point cloud to be used to fit the clustering
            model.
        :return: The clusterer itself, for fluent programming purposes.
        :rtype: :class:`.Clusteror`
        """
        return self

    @abstractmethod
    def cluster(self, pcloud):
        """
        Clustering from a given input point cloud.

        :param pcloud: The input point cloud for which clusters must be found.
        :return: The point cloud extended with the clusters.
        :rtype: :class:`.PointCloud`
        """
        pass

    def post_process(self, pcloud):
        """
        Run the post-processing pipeline on the given input point cloud.

        :param pcloud: The input point cloud for the components in the
            post-processing pipeline.
        :type pcloud: :class:`.PointCloud`
        :return: The post-processed point cloud. Sometimes it will be exactly
            the same input point cloud because some post-processing components
            generate their output directly to a file.
        :rtype: :class:`.PointCloud`
        """
        # Ignore empty post-processing pipelines
        if self.post_clustering is None or len(self.post_clustering) < 1:
            return pcloud
        # Measure start time
        start = time.perf_counter()
        # Apply post-processing pipeline
        for callable in self.post_clustering:  # For each callable in the pipe.
            callable_start = time.perf_counter()
            pcloud = callable(self, pcloud, out_prefix=self.out_prefix)  # Run
            callable_end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'{self.__class__.__name__} computed '
                f'{callable.__class__.__name__} on {pcloud.get_num_points()} '
                f'points in {callable_end-callable_start:.3f} seconds.'
            )
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'{self.__class__.__name__} post-processed '
            f'{pcloud.get_num_points()} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return pcloud

    def fit_cluster_and_post_process(self, pcloud, out_prefix=None):
        """
        Compute the fitting, clustering, and post-processing as a whole.

        See :meth:`.Clusterer.fit`, :meth:`.Clusterer.cluster`, and
        :meth:`Clusterer.post_process`.

        :param pcloud: The input point cloud to be used to fit the clustering
            model.
        :type pcloud: :class:`.PointCloud`
        :param out_prefix: If given, it will be used to replace the default
            output prefix of the clusterer inside the call's context.
        :type out_prefix: str or None
        :return: The point cloud extended with the clusters.
        :rtype: :class:`.PointCloud`
        """
        # Update output prefix
        if out_prefix is not None:
            _out_prefix = self.out_prefix
            self.out_prefix = out_prefix
        exception = None
        # Fit, cluster and post-process
        try:
            self.fit(pcloud)
            pcloud = self.cluster(pcloud)
            pcloud = self.post_process(pcloud)
        except Exception as ex:
            exception = ex
        # Restore output prefix
        if out_prefix is not None:
            self.out_prefix = _out_prefix
        # Return
        if exception is not None:
            raise exception
        return pcloud

    def add_cluster_labels_to_point_cloud(self, pcloud, c):
        """
        Add given cluster labels to the given point cloud.

        :param pcloud: The point cloud to which the cluster labels must be
            added.
        :type pcloud: :class:`.PointCloud`
        :param c: The cluster labels.
        :type c: :class:`np.ndarray`
        :return: The input point cloud (updated inplace to add the cluster
            labels).
        :rtype: :class:`.PointCloud`
        """
        return pcloud.add_features(
            [self.cluster_name], c.reshape(-1, 1), ftypes=['d']
        )
