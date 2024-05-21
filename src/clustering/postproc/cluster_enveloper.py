# ---   IMPORTS   --- #
# ------------------- #
from src.clustering.postproc.clustering_post_processor import \
    ClusteringPostProcessor, ClusteringException
import src.main.main_logger as LOGGING
import open3d
import numpy as np
import time


# ---   CLASS   --- #
# ⨪---------------- #
class ClusterEnveloper(ClusteringPostProcessor):
    """
    :author: Alberto M. Esmoris Pena

    Clustering post-processor that computes the requested envelopes for each
    cluster.
    See :class:`.ClusteringPostProcessor`
    """
    # TODO Rethink : Doc ivars
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a ClusterEnveloper post-processor.

        See :meth:`.ClusteringPostProcessor.__init__`.

        :param kwargs: The key-word arguments for the initialization of the
            ClustererEnveloper.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.envelopes = kwargs.get('envelopes')

    # ---  POST-PROCESSING CALL  --- #
    # ------------------------------ #
    def __call__(self, clusterer, pcloud, out_prefix=None):
        """
        Post-process the given point cloud with clusters to compute the
        cluster-wise envelopes.

        :param clusterer: The clusterer that generated the clusters.
        :type clusterer: :class:`.Clusterer`
        :param pcloud: The point cloud to be post-processed.
        :type pcloud: :class:`.PointCloud`
        :param out_prefix: The output prefix in case path expansion must be
            applied.
        :type out_prefix: str or None
        :return: The post-processed point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Prepare post-processing
        start = time.perf_counter()
        X = pcloud.get_coordinates_matrix()
        c = self.get_cluster_labels(clusterer, pcloud)
        c_dom = np.unique(c[c > -1])  # All unique values but noise (-1)
        # Compute envelops for each cluster
        for ck in c_dom:
            I = c == ck  # Indices of points in the cluster
            for envelope in self.envelopes:
                envelope_type = envelope['type']
                type_low = envelope_type.lower()
                if type_low in ['aabb', 'alignedboundingbox']:
                    self.compute_aabb_envelope(envelope, X[I], ck)
                elif type_low in ['bbox', 'boundingbox']:
                    self.compute_bbox_envelope(envelope, X[I], ck)
                else:
                    LOGGING.LOGGER.error(
                        'ClusterEnveloper received an unexpected envelope: '
                        f'"{envelope_type}"'
                    )
                    raise ClusteringException(
                        'ClusterEnveloper received an unexpected envelope: '
                        f'"{envelope_type}"'
                    )
        # Export envelope information
        if getattr(self, "aabbs", None) is not None:
            self.export_aabb_envelopes(out_prefix=out_prefix)
        if getattr(self, "bboxes", None) is not None:
            self.export_bbox_envelopes(out_prefix=out_prefix)
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClusterEnveloper finished in {end-start:.3f} seconds.'
        )
        # Return
        return pcloud

    # ---  ENVELOPE COMPUTATION METHODS  --- #
    # -------------------------------------- #
    def compute_aabb_envelope(self, spec, X, cidx):
        """
        Compute the axis-aligned bounding box for the given points.

        :param spec: The specification of the axis aligned bounding box to
            be computed.
        :type spec: dict
        :param X: The structure space matrix representing the cluster whose
            axis-aligned bounding box must be found.
        :type X: :class:`np.ndarray`
        :param cidx: The index of the cluster whose envelope must be computed.
        :type cidx: int
        :return: Nothing, but the generated axis-aligned bounding box is stored
            in the member attribute ``aabbs``.
        """
        # Make sure path is stored for future exports
        if getattr(self, "aabbs_path", None) is None:
            self.aabbs_path = spec.get('output_path', None)
        # Make sure separator is set
        if getattr(self, 'aabbs_separator', None) is None:
            self.aabbs_separator = spec.get('separator', None)
        # Make sure member attributes aabbs exist
        if getattr(self, "aabbs", None) is None:
            self.aabbs = []
        # Compute bounding box vertices
        aabb = {}
        a, b = np.min(X, axis=0), np.max(X, axis=0)  # Min and max vertices
        aabb['vertices'] = np.array([  # The eight vertices, explicitly
            [x, y, z]
            for x in [a[0], b[0]]
            for y in [a[1], b[1]]
            for z in [a[2], b[2]]
        ])
        aabb['cluster_label'] = cidx
        # Compute enclosed volume, if requested
        if spec.get('compute_volume', False):
            vol = np.prod(np.abs(b-a))  # Volume
            aabb['volume'] = vol
        # Store computed bounding box
        self.aabbs.append(aabb)

    def compute_bbox_envelope(self, spec, X, cidx):
        """
        Compute the bounding box for the given points.

        :param spec: The specification of the bounding box to be computed.
        :type spec: dict
        :param X: The structure space matrix representing the cluster whose
            bounding box must be found.
        :type X: :class:`np.ndarray`
        :param cidx: The index of the cluster whose envelope must be computed.
        :type cidx: int
        :return: Nothing, but the generated bounding box is stored in
            the member attribute ``bboxes``.
        """
        # Make sure path is stored for future exports
        if getattr(self, "bboxes_path", None) is None:
            self.bboxes_path = spec.get('output_path', None)
        # Make sure separator is set
        if getattr(self, 'bboxes_separator', None) is None:
            self.bboxes_separator = spec.get('separator', None)
        # Make sure member attributes bboxes exist
        if getattr(self, "bboxes", None) is None:
            self.bboxes = []
        # Compute bounding box vertices
        bbox = {}
        points = open3d.utility.Vector3dVector(X)
        o3d_bbox = open3d.geometry.OrientedBoundingBox.create_from_points(points)
        bbox['vertices'] = np.asarray(o3d_bbox.get_box_points())
        bbox['cluster_label'] = cidx
        # Compute enclosed volume, if requested
        if spec.get('compute_volume', False):
            bbox['volume'] = o3d_bbox.volume()
        # Store computed bounding box
        self.bboxes.append(bbox)

    # ---  ENVELOPE EXPORT METHODS  --- #
    # --------------------------------- #
    def export_aabb_envelopes(self, out_prefix=None):
        """
        Export the axis-aligned bounding box envelopes stored as member
        attributes.

        :return: Nothing at all, but the bounding boxes are written to files.
        """
        # Prepare output path
        outpath = self.aabbs_path
        if out_prefix is not None:
            outpath = out_prefix[:-1] + outpath[1:]
        # Prepare output format
        sep = self.aabbs_separator
        cluster_label = 'cluster_label' in self.aabbs[0]
        volume = 'volume' in self.aabbs[0]
        # Write bounding boxes
        with open(outpath, 'w') as outf:
            for aabb in self.aabbs:
                s = ''
                for vertex in aabb['vertices']:
                    s += f'{vertex[0]:.4f}{sep}{vertex[1]:.4f}{sep}{vertex[2]:.4f}'
                    if cluster_label:
                        s += f'{sep}{int(aabb["cluster_label"]):d}'
                    if volume:
                        s += f'{sep}{aabb["volume"]:.4f}'
                    s += '\n'
                outf.write(s)
        # Report output path
        LOGGING.LOGGER.info(
            f'ClusterEnveloper exported {len(self.aabbs)} axis-aligned '
            f'bounding boxes to "{outpath}".'
        )

    def export_bbox_envelopes(self, out_prefix=None):
        """
        Export the bounding box envelopes stored as member attributes.

        :return: Nothing at all, but the bounding boxes are written to files.
        """
        # Prepare output path
        outpath = self.bboxes_path
        if out_prefix is not None:
            outpath = out_prefix[:-1] + outpath[1:]
        # Prepare output format
        sep = self.bboxes_separator
        cluster_label = 'cluster_label' in self.bboxes[0]
        volume = 'volume' in self.bboxes[0]
        # Write bounding boxes
        with open(outpath, 'w') as outf:
            for bbox in self.bboxes:
                s = ''
                for vertex in bbox['vertices']:
                    s += f'{vertex[0]:.4f}{sep}{vertex[1]:.4f}{sep}{vertex[2]:.4f}'
                    if cluster_label:
                        s += f'{sep}{int(bbox["cluster_label"]):d}'
                    if volume:
                        s += f'{sep}{bbox["volume"]:.4f}'
                    s += '\n'
                outf.write(s)
        # Report output path
        LOGGING.LOGGER.info(
            f'ClusterEnveloper exported {len(self.bboxes)} bounding boxes '
            f'to "{self.bboxes_path}"'
        )


