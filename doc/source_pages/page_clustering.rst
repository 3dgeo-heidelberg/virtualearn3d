.. _Clustering page:

Clustering
***************

Clustering consists of receiving an input point cloud and finding points that
can be grouped together into clusters. The clusters are typically represented
with integer values :math:`c \in \mathbb{Z}_{\geq -1}`, where the value
:math:`-1` is used to represent no cluster or noise cluster. Clustering
computations are carried out through :class:`.Clusterer` components. They
can be included in pipelines. Besides, it is possible to define a
:ref:`post-processing pipeline <Clustering post-processing>`
that will be computed immediately after the clustering.




Clusterers
=============



.. _DBScan clusterer:

DBScan clusterer
-------------------

The :class:`.DBScanClusterer` can be used to compute Density-Based Spatial
Clustering of Applications with Noise (DBSCAN) in the point cloud. The DBSCAN
algorithm considers a minimum number of points and a radius to define the
neighborhood for spatial queries. It finds kernel points, those in whose
neighborhood there are at least the minimum number of points. Then, it
considers any accessible point to be in the cluster. The accessible points
that are not kernel points are included in the cluster because they are
close enough to the kernel points. However, these points are not considered
to include new points as only kernel points can be used to determine whether
a point is accessible or not. The JSON below shows how to define a DBScan
clustering component:


.. code-block:: json

    {
        "clustering": "fpsdecorated",
        "fps_decorator": {
            "num_points": "m/10",
            "fast": true,
            "num_encoding_neighbors": 1,
            "num_decoding_neighbors": 1,
            "release_encoding_neighborhoods": false,
            "threads": 16,
            "representation_report_path": null
        },
        "decorated_clusterer": {
            "clustering": "dbscan",
            "cluster_name": "cluster_wood",
            "precluster_name": "Prediction",
            "precluster_domain": [1],
            "min_points": 200,
            "radius": 3.0,
            "post_clustering": [
                {
                    "post-processor": "ClusterEnveloper",
                    "envelopes": [
                        {
                            "type": "AlignedBoundingBox",
                            "compute_volume": true,
                            "output_path": "*/aabbs.csv",
                            "separator": ","
                        },
                        {
                            "type": "BoundingBox",
                            "compute_volume": true,
                            "output_path": "*/bboxes.csv",
                            "separator": ","
                        }
                    ]
                }
            ]
        }
    }


The JSON above defines a :class:`.DBScanClusterer` that is decorated to work
on a FPS representation of the input point cloud (see
:ref:`the FPS decorated clusterer documentation<FPS decorated clusterer>`
for further details). It will consider all the points predicted as wood to
compute clusters of wood. The minimum number of points in the neighborhoods of
kernel points is set to 200 points, and the radius of the neighborhood to
3 meters. After computing the clustering, a post-processing pipeline will be
run to find the axis-aligned and oriented bounding boxes, respectively. For
each bounding box the vertices, the cluster label, and the enclosed volume will
be exported to a CSV file using ``,`` as separator.

**Arguments**

-- ``cluster_name``
    The name of the clusters. For instance, when finding clusters of
    buildings it is a good idea to use ``"cluster_building"`` as the
    cluster name.


-- ``precluster_name``
    In case the point cloud is already clustered, the precluster name
    specifies the point-wise attribute to consider to filter the points.
    Note that both reference and predicted classes can be seen as clusters.


-- ``precluster_domain``
    If the preclustering filter is applied, then the precluster domain list
    can be used to specify the values of interest (any point that is not
    in the domain will be ignored). If not given and preclustering filter
    is requested, then the precluster domain will be automatically inferred
    as the set of unique values for the attribute.


-- ``min_points``
    The minimum number of points that must belong to a neighborhood to
    pass the kernel point check.


-- ``radius``
    The radius of the neighborhood for kernel point checks.


-- ``post_clustering``
    The post-clustering pipeline. It can be ``null`` or a list of
    :ref:`post-clustering components <Clustering post-processing>`.








.. _Clustering post-processing:

Post-processing
==================

Once clusters are available, it might be useful to apply some post-processing
to derive useful information from them. That is what post-processing
components aim to do. They can be defined in the ``"post_clustering"`` list.
The execution will start by the first component in the list and will end
at the last one, i.e., it is sequential with respect to the order in the list.


ClusterEnveloper
--------------------

The :class:`.ClusterEnveloper` can be used to find the geometry enclosing the
points. Envelopes often support the estimation of the enclosed volume. They can
be exported to output files. The JSON below shows how to define a cluster
enveloper post-processor inside the post-processing pipeline of a clusterer.


.. code-block:: json

    "post_clustering": [
        {
            "post-processor": "ClusterEnveloper",
            "envelopes": [
                {
                    "type": "AlignedBoundingBox",
                    "compute_volume": true,
                    "output_path": "*/aabbs.csv",
                    "separator": ","
                },
                {
                    "type": "BoundingBox",
                    "compute_volume": true,
                    "output_path": "*/bboxes.csv",
                    "separator": ","
                }
            ]
        }
    ]


The JSON above defines a :class:`.ClustererEnveloper` that will compute the
axis-aligned bounding box and also the oriented bounding box. The volume will
be computed for each envelope and they will be exported to a CSV file.
More concretely, each row in the CSV will contain three coordinates
representing a vertex, an integer representing the cluster label, and the
estimated volume. All the values will be separated by ``,``.


**Arguments**

-- ``envelopes``
    List of envelops that must be computed by the :class:`.ClustererEnveloper`

    -- ``type``

    The type of the envelope, it can be ``"AxisAlignedBoundingBox"`` or
    ``"BoundingBox"`` (the edges of the last one are not necessarily parallel
    to the vectors of the canonical basis).

    -- ``compute_volume``

    Flag to specify whether to compute the volume enclosed by the envelope
    (``true``) or not (``false``).

    -- ``output_path``

    The path where the envelope's data will be exported (typically in CSV
    format).

    -- ``separator``

    The separator to be used in the output CSV file.









.. _FPS decorated clusterer:

Decorators
===============

Furthest point sampling decorator
--------------------------------------------

The :class:`.FPSDecoratedClusterer` can be used to decorate a data miner such
that the computations can take place in a transformed space of reduced
dimensionality. Typically, the domain of a clustering is the entire point
cloud, let us say :math:`m` points. When using a :class:`.FPSDecoratedClusterer`
this domain will be transformed to a subset of the original point cloud with
:math:`R` points, such that :math:`m \geq R`. Decorating a clusterer with this
decorator can be useful to reduce its execution time.


.. code-block:: json

    {
        "clustering": "fpsdecorated",
        "fps_decorator": {
            "num_points": "m/10",
            "fast": true,
            "num_encoding_neighbors": 1,
            "num_decoding_neighbors": 1,
            "release_encoding_neighborhoods": false,
            "threads": 16,
            "representation_report_path": null
        },
        "decorated_clusterer": {
            "clustering": "dbscan",
            "cluster_name": "cluster_wood",
            "precluster_name": "Prediction",
            "precluster_domain": [1],
            "min_points": 200,
            "radius": 3.0,
            "post_clustering": [
                {
                    "post-processor": "ClusterEnveloper",
                    "envelopes": [
                        {
                            "type": "AlignedBoundingBox",
                            "compute_volume": true,
                            "output_path": "*/aabbs.csv",
                            "separator": ","
                        },
                        {
                            "type": "BoundingBox",
                            "compute_volume": true,
                            "output_path": "*/bboxes.csv",
                            "separator": ","
                        }
                    ]
                }
            ]
        }
    }

**Arguments**

-- ``fps_decorator``
    The specification of the furthest point sampling (FPS) decoration carried
    out through the :class:`.FPSDecoratorTransformer`.

    -- ``num_points``
        The target number of points :math:`R` for the transformed point cloud.
        It can be an integer or an expression that will be evaluated with
        :math:`m` representing the number of points of the original point
        cloud, e.g., ``"m/2"`` will downscale the point cloud to half the
        number of points.

    -- ``fast``
        Whether to use exact furthest point sampling (``false``) or a faster
        stochastic approximation (``true``).

    -- ``num_encoding_neighbors``
        How many closest neighbors in the original point cloud are considered
        for each point in the transformed point cloud to reduce from the
        original space to the transformed one.

    -- ``num_decoding_neighbors``
        How many closest neighbors in the transformed point cloud are
        considered for each point in the original point cloud to propagate back
        from the transformed space to the original one.

    -- ``release_encoding_neighborhoods``
        Whether the encoding neighborhoods can be released after computing the
        transformation (``true``) or not (``false``). Releasing these
        neighborhoods means the :meth:`.FPSDecoratorTransformer.reduce` method
        must not be called, otherwise errors will arise. Setting this flag to
        true can help saving memory when needed.

    -- ``threads``
        The number of parallel threads to consider for the parallel
        computations. Note that ``-1`` means using as many threads as available
        cores.

    -- ``representation_report_path``
        Where to export the transformed point cloud. In general, it should be
        ``null`` to prevent unnecessary operations. However, it can be enabled
        (by given any valid path to write a point cloud file) to visualize the
        points that are seen by the data miner.

-- ``decorated_clusterer``
    A typical clustering specification. See
    :ref:`the DBScan clusterer <DBScan clusterer>` for an example.
