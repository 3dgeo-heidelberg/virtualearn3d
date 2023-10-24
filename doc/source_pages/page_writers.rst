.. _Writers page:

Writers
*********

Writers are one of the many components supported by the VirtuaLearn3D (VL3D)
framework. They are especially useful to design custom pipelines because they
allow to write data from the state variables (e.g., the point cloud) at
different moments during execution. In this section, the many available writers
are commented. Readers are strongly encouraged to get familiar with the idea
of pipelines (see :ref:`page on pipelines <Pipelines page>`).




Writer
=========

The :class:`.Writer` is the most simple writer in the framework. It takes the
point cloud at the current pipeline's state and exports it as it is. A
:class:`.Writer` can be defined inside a pipeline using the JSON below:

.. code-block:: json

    {
        "writer": "Writer",
        "out_pcloud": "my_pointcloud.laz"
    }

The JSON above defines a :class:`.Writer` that will export the point cloud to a
LAZ file called `my_pointcloud.laz`.




Model writer
==============

The :class:`.ModelWriter` does what its name says. It exports the model as it
is at the current pipeline's state. The recommendation is to write the models
to files ending with a `.model` extension. A :class:`.ModelWriter` can be
defined inside a pipeline using the JSON below:

.. code-block:: json

    {
        "writer": "ModelWriter",
        "out_model": "my_random_forest.model"
    }

The JSON above defines a :class:`.Writer` that will export the model to a
file called `my_random_forest.model`.




Predictions writer
====================

The :class:`.PredictionsWriter` does what its name says. It exports the
available predictions at the current pipeline's state. Typically, the
predictions are exported to text files as a column of labels. A
:class:`.PredictionsWriter` can be defined inside a pipeline using the JSON
below:

.. code-block:: json

    {
        "writer": "PredictionsWriter",
        "out_preds": "my_predictions.lbl"
    }

The JSON above defines a :class:`.PredictionsWriter` that will export the
predictions to a file called `my_predictions.lbl`.




Classified point cloud writer
===============================

The :class:`.ClassifiedPcloudWriter` does more stuff than the baseline
:class:`.PredictionsWriter`. More concretely, it analyzes
the pipeline's state to generate a state-dependent output. If there are
predictions available, they will be incorporated to the point cloud and
exported as a LAS/LAZ file for further visualization. On top of that, if the
classification labels are available in the point cloud, a new attribute
called `Success` will be added. This new attribute takes a value of one
when the prediction matches the reference label and zero when it does not. A
:class:`.ClassifiedPcloudWriter` can be defined inside a pipeline using the
JSON below:

.. code-block:: json

    {
        "writer": "ClassifiedPcloudWriter",
        "out_pcloud": "my_classified_pcloud.laz"
    }

The JSON above defines a :class:`.ClassifiedPcloudWriter` that will export the
classified point cloud to a file called `my_classified_pcloud.laz`.




Predictive pipeline writer
============================

The :class:`.PredictivePipelineWriter` is meant to be used in sequential
pipelines that train a model to export the model together with the many
components in the pipeline that are necessary to reproduce the predictions,
e.g., feature transformation or data mining components. A
:class:`.PredictivePipelineWriter` can be defined inside a pipeline using
the JSON below. For better understanding, readers are referred to the
documentation of :ref:`predictive pipelines <Predictive pipeline section>`.


.. code-block:: json

    {
        "writer": "PredictivePipelineWriter",
        "out_pipeline": "my_pwise_classif.pipe",
        "include_writer": false,
        "include_imputer": true,
        "include_feature_transformer": true,
        "include_miner": true,
        "include_class_transformer": true
    }

The JSON above defines a :class:`.PredictivePipelineWriter` that will export
a sequential pipeline transformed to a predictive pipeline. The pipeline
will be stored in a file called `my_pwise_classif.pipe`. The boolean flags
govern what components are exported together with the model. For instance,
setting a ``include_imputer`` to true implies any imputation strategy in the
pipeline will be exported together with the model (in the same sequential
order). However, setting it to false implies imputation strategies considered
during training will not be part of the predictive pipeline.

