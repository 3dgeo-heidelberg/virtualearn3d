# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.clustering.clusterer import ClusteringException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClusteringPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Interface governing any component of a clustering post-processing pipeline.
    See :class:`.Clusterer` and :meth:`.Clusterer.post_process`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a ClusteringPostProcessor.

        :param kwargs: The key-word arguments for the initialization of any
            ClusteringPostProcessor.
        """
        pass

    # ---  POST-PROCESSING CALL  --- #
    # ------------------------------ #
    @abstractmethod
    def __call__(self, clusterer, pcloud):
        """
        Abstract method that must be overridden by any concrete (instantiable)
        component of a clustering post-processing pipeline.

        :param clusterer: The clusterer that called the post-processor.
        :param pcloud: The point cloud that must be post-processed.
        :return: The post-processed point cloud.
        :rtype: :class:`.PointCloud`
        """
        pass

    # ---   BUILD POST-PROCESSORS   --- #
    # --------------------------------- #
    @staticmethod
    def build_post_processor(spec):
        """
        Build the post-processor from its key-words specification.

        :param spec: The post-processor specification.
        :type spec: dict
        :return: Built post-processor.
        :rtype: callable
        """
        # Prepare post-processor specification
        processor = spec.get('post-processor', None)
        processor_low = processor.lower()
        processor_spec = dict(spec)
        del processor_spec['post-processor']
        # Build expected post-processors
        if processor_low == 'clusterenveloper':
            from src.clustering.postproc.cluster_enveloper \
                import ClusterEnveloper
            return ClusterEnveloper(**processor_spec)
        # Exception for unexpected post-processors
        raise ClusteringException(
            f'Unexpected post-processor: "{processor}"'
        )

    # ---  CLUSTERING POST-PROCESSING UTILS  --- #
    # ------------------------------------------ #
    def get_cluster_labels(self, clusterer, pcloud):
        """
        Obtain the vector of cluster labels corresponding to the given
        clusterer.

        :param clusterer: The clusterer whose labels must be extracted.
        :type clusterer: :Class:`.Clusterer`
        :param pcloud: The clustered point cloud.
        :type pcloud: :class:`.PointCloud`
        :return: The vector of point-wise cluster labels.
        :rtype: :class:`np.ndarray`
        """
        # Get deepest clusterer
        final_clusterer = clusterer
        while hasattr(final_clusterer, 'decorated_clusterer'):
            final_clusterer = clusterer.deocrated_clusterer
        # Return point-wise cluster labels
        return pcloud.get_features_matrix([
            final_clusterer.cluster_name
        ]).flatten()
