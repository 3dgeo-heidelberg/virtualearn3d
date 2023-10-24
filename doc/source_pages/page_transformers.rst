.. _Transformers page:

Transformers
****************

Transformers are components that apply a transformation on the point cloud.
They can be divided into class transformers (:class:`.ClassTransformer`) that
transform the classification and predictions of the point cloud, feature
transformers (:class:`.FeatureTransformer`) that transform the features of
the point cloud, and point transformers (:class:`.PointTransformers`) that
compute an advanced transformation on the point cloud that involves different
information (e.g., spatial coordinates to derive
receptive fields that can be used to reduce or propagate both features and
classes).

Transformers are typically use inside pipelines to apply transformations to
the point cloud at the current pipeline's state. Readers are strongly
encouraged to read the :ref:`Pipelines documentation<Pipelines page>` before
looking further into transformers.



Class transformers
====================


Class reducer
---------------

The :class:`.ClassReducer` takes an original set of :math:`n_I` input classes
and returns :math:`n_O` output classes, where :math:`n_O < n_I`. It can be
applied to the reference classification only or also to the predictions.
On top of that, it supports a text report on the distributions with the
absolute and relative frequencies and a plot of the class distribution before
and after the transformation. A :class:`.ClassReducer` can be defined inside a
pipeline using the JSON below:

.. code-block:: json

	{
		"class_transformer": "ClassReducer",
		"on_predictions": false,
		"input_class_names": ["noclass", "ground", "vegetation", "cars", "trucks", "powerlines", "fences", "poles", "buildings"],
		"output_class_names": ["noclass", "ground", "vegetation", "buildings", "objects"],
		"class_groups": [["noclass"], ["ground"], ["vegetation"], ["buildings"], ["cars", "trucks", "powerlines", "fences", "poles"]],
		"report_path": "class_reduction.log",
		"plot_path": "class_reduction.svg"
	}

The JSON above defines a :class:`.ClassReducer` that will replace the nine
original classes into five reduced classes where many classes are grouped
together as the ``"objects"`` class. Moreover, it will generate a text report
in a file called `class_reduction.log` and a figure representing the class
distribution in `class_reduction.svg`.


**Arguments**

-- ``on_predictions``
	Whether to also reduce the predictions if any (True) or not (False). Note
	that setting ``on_predictions`` to True will only work if there are
	available predictions.

-- ``input_class_names``
	A list with the names of the input classes.

-- ``output_class_names``
	A list with the desired names for the output classes.

-- ``class_groups``
	A list of lists such that the list i defines which classes will be
	considered to obtain the reduced class i. In other words, each sublist
	contains the strings representing the names of the input classes that
	must be mapped to the output class.

-- ``report_path``
	Path where the text report on the class distributions must be written. If
	it is not given, then no report will be generated.

-- ``plot_path``
	Path where the plot of the class distributions must be written. If it is
	not given, then no plot will be generated.


**Output**

The examples in this section come from applying a :class:`.ClassReducer` to the
`5080_54435.laz` point cloud of the
`DALES dataset <https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php>`_
.

An example of the plot representing how the classes are distributed
before and after the :class:`.ClassReducer` is shown below.

.. figure:: ../img/class_reducer_plot.png
	:scale: 15%
	:alt:   Figure representing the distribution of classes before and after
			the class reduction

	Visualization of the class distributions before and after the class
	reduction.


An example of how the classes represented on the point cloud look like before
and after the :class:`.ClassReducer` is shown below.

.. figure:: ../img/class_reducer_pcloud.png
	:scale: 33%
	:alt: Figure representing a class reduction.

	Visualization of the original (left) and reduced classification (right).








Feature transformers
=======================


Minmax normalizer
-------------------

.. code-block:: json

    {
        "feature_transformer": "MinmaxNormalizer",
        "fnames": ["AUTO"],
        "target_range": [0, 1],
        "clip": true,
        "report_path": "minmax_normalization.log"
    }


**Arguments**


**Output**

A transformed point cloud is generated such that its features are normalized
to a [0, 1] interval. The min, the max, and the range are exported through
the logging system (see below for an example corresponding to the minmax
normalization of some geometric features).

.. list-table::
    :widths: 31 23 23 23
    :header-rows: 1

    *   - FEATURE
        - MIN
        - MAX
        - RANGE
    *   - linearity_r0.05
        - 0.00028
        - 1.00000
        - 0.99972
    *   - planarity_r0.05
        - 0.00000
        - 0.97660
        - 0.97660
    *   - surface_variation_r0.05
        - 0.00000
        - 0.32316
        - 0.32316
    *   - eigenentropy_r0.05
        - 0.00006
        - 0.01507
        - 0.01501
    *   - omnivariance_r0.05
        - 0.00000
        - 0.00060
        - 0.00060
    *   - verticality_r0.05
        - 0.00000
        - 1.00000
        - 1.00000
    *   - anisotropy_r0.05
        - 0.06250
        - 1.00000
        - 0.93750
    *   - linearity_r0.1
        - 0.00070
        - 1.00000
        - 0.99930
    *   - planarity_r0.1
        - 0.00000
        - 0.95717
        - 0.95717
    *   - surface_variation_r0.1
        - 0.00000
        - 0.32569
        - 0.32569
    *   - eigenentropy_r0.1
        - 0.00028
        - 0.04501
        - 0.04473
    *   - omnivariance_r0.1
        - 0.00000
        - 0.00241
        - 0.00241
    *   - verticality_r0.1
        - 0.00000
        - 1.00000
        - 1.00000
    *   - anisotropy_r0.1
        - 0.05643
        - 1.00000
        - 0.94357



Standardizer
--------------

.. code-block:: json

    {
        "feature_transformer": "Standardizer",
        "fnames": ["AUTO"],
        "center": true,
        "scale": true,
        "report_path": "standardization.log"
    }


**Arguments**

**Output**

A transformed point cloud is generated such that its features are
standardized. The mean and standard deviation are exported through the
logging system (see below for an example corresponding to the standardization
of some geometric features).

.. list-table::
    :widths: 40 30 30
    :header-rows: 1

    *   - FEATURE
        - MEAN
        - STDEV.
    *   - linearity_r0.05
        - 0.47259
        - 0.24131
    *   - planarity_r0.05
        - 0.32929
        - 0.22213
    *   - surface_variation_r0.05
        - 0.10697
        - 0.06362
    *   - eigenentropy_r0.05
        - 0.00781
        - 0.00184
    *   - omnivariance_r0.05
        - 0.00025
        - 0.00010
    *   - verticality_r0.05
        - 0.55554
        - 0.30274
    *   - anisotropy_r0.05
        - 0.80188
        - 0.14316
    *   - linearity_r0.1
        - 0.49389
        - 0.24075
    *   - planarity_r0.1
        - 0.29196
        - 0.21008
    *   - surface_variation_r0.1
        - 0.11583
        - 0.06376
    *   - eigenentropy_r0.1
        - 0.02512
        - 0.00533
    *   - omnivariance_r0.1
        - 0.00100
        - 0.00035
    *   - verticality_r0.1
        - 0.57260
        - 0.30121
    *   - anisotropy_r0.1
        - 0.78585
        - 0.14570



Variance selector
--------------------

.. code-block:: json

    {
        "feature_transformer": "VarianceSelector",
        "fnames": ["AUTO"],
        "variance_threshold": 0.01,
        "report_path": "variance_selection.log"
    }

**Arguments**

**Output**

.. list-table::
    :widths: 60 40
    :header-rows: 1

    *   - FEATURE
        - VARIANCE
    *   - omnivariance_r0.05
        - 0.000
    *   - omnivariance_r0.1
        - 0.000
    *   - eigenentropy_r0.05
        - 0.000
    *   - eigenentropy_r0.1
        - 0.000
    *   - surface_variation_r0.05
        - 0.004
    *   - surface_variation_r0.1
        - 0.005
    *   - anisotropy_r0.05
        - 0.020
    *   - anisotropy_r0.1
        - 0.022
    *   - linearity_r0.1
        - 0.051
    *   - linearity_r0.05
        - 0.056
    *   - planarity_r0.1
        - 0.066
    *   - planarity_r0.05
        - 0.075
    *   - verticality_r0.05
        - 0.092
    *   - verticality_r0.1
        - 0.097

.. list-table::
    :widths: 100
    :header-rows: 1

    *   - SELECTED FEATURES
    *   - linearity_r0.05
    *   - planarity_r0.05
    *   - verticality_r0.05
    *   - anisotropy_r0.05
    *   - linearity_r0.1
    *   - planarity_r0.1
    *   - verticality_r0.1
    *   - anisotropy_r0.1





K-Best selector
------------------

.. code-block:: json

    {
        "feature_transformer": "KBestSelector",
        "type": "classification",
        "k": 2,
        "fnames": ["AUTO"],
        "report_path": "kbest_selection.log"
    }

**Arguments**

**Output**

.. csv-table::
    :file: ../csv/kbest_selector_report.csv
    :widths: 40 30 30
    :header-rows: 1


.. list-table::
    :widths: 100
    :header-rows: 1

    *   - SELECTED FEATURES
    *   - surface_variation_r0.1
    *   - anisotropy_r0.1


Percentile selector
----------------------

.. code-block:: json

    {
	  "feature_transformer": "PercentileSelector",
	  "type": "classification",
	  "percentile": 20,
	  "fnames": ["AUTO"],
	  "report_path": "*report/kbest_selection.log"
	}

**Arguments**

**Output**


.. csv-table::
    :file: ../csv/percentile_selector_report.csv
    :widths: 40 30 30
    :header-rows: 1

.. list-table::
    :widths: 100
    :header-rows: 1

    *   - SELECTED FEATURES
    *   - surface_variation_r0.1
    *   - verticality_r0.1
    *   - anisotropy_r0.1


PCA transformer
------------------

.. code-block:: json

    {
        "feature_transformer": "PCATransformer",
        "out_dim": 0.99,
        "whiten": false,
        "random_seed": null,
        "fnames": ["AUTO"],
        "report_path": "pca_projection.log",
        "plot_path": "pca_projection.svg"
    }

**Arguments**

**Output**

.. csv-table::
    :file: ../csv/pca_transformer_report.csv
    :widths: 20 20
    :header-rows: 1

.. figure:: ../img/pca_transformer_plot.png
    :scale: 20%
    :alt: Figure representing the PCA-derived features by aggregated explained
          variance.

    The relationship between the PCA-derived features and the aggregated
    explained variance ratio.

.. figure:: ../img/pca_transformer_comparison.png
    :scale: 30%
    :alt:   Figure representing three different features that have been reduced
            to a single one using PCA.

    The anisotropy, surface variation, and verticality computed for spherical
    neighborhoods with :math:`10\,\mathrm{cm}` radius reduced to a single
    feature through PCA.







Point transformers
====================

At the moment, point transformers are used in the context of deep learning
models. They are not available as independent components for pipelines.