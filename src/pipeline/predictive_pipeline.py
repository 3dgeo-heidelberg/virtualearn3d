# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pipeline import Pipeline, PipelineException


# ---   CLASS   --- #
# ----------------- #
class PredictivePipeline(Pipeline):
    r"""
    :author: Alberto M. Esmoris Pena

    A predictive pipeline is any pipeline that can be used as an estimator.

    In other words, the predictive pipeline can be seen as a map :math:`f`
    from a given input :math:`x` that yields the corresponding estimations,
    aiming to approximate as much as possible the actual values :math:`y`.

    More formally:

    .. math::
        f(x) \approx y

    However, the predictive pipeline itself is not limited to the predictive
    model :math:`\hat{y}`. It also accounts for other components such as the
    data mining, imputation, and feature transformation.

    For instance, let :math:`m_1` represent a data miner, :math:`m_2` another
    data miner, and :math:`i` represent a data imputer. For this case, the
    composition of these components with the estimator :math:`\hat{y}` would
    lead to a sequential predictive pipeline that can be described as follows:

    .. math::
        f(x) = (\hat{y} \circ i \circ m_2 \circ m_1)(x)

    :ivar pipeline: The wrapped pipeline. It must be possible to use it to
        compute predictions. For example, a pipeline made of data mining
        components only will fail.
    :vartype pipeline: `.Pipeline`
    :ivar pps: The pipeline's predictive strategy. It must be compatible with
        the wrapped pipeline. The strategy defines how to use the pipeline
        to make predictions.
    :vartype pps: `.PipelinePredictiveStrategy`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, pipeline, pps, **kwargs):
        # Call parent's init
        super().__init__(
            in_pcloud=kwargs.get('in_pcloud', pipeline.in_pcloud),
            out_pcloud=kwargs.get('out_pcloud', pipeline.out_pcloud)
        )
        # Assign attributes
        self.pipeline = pipeline
        self.pps = pps
        # Validate
        if self.pipeline is None:
            raise PipelineException(
                'PredictivePipeline requires a pipeline. None was given.'
            )
        if self.pps is None:
            raise PipelineException(
                'PredictivePipeline requires a PipelinePredictiveStrategy. '
                'None was given.'
            )

    # ---  PREDICTIVE PIPELINE METHODS  --- #
    # ------------------------------------- #
    def predict(self, pcloud):
        """
        The predict method computes the predictions from the wrapped pipeline.

        :param pcloud: The point cloud to be predicted.
        :type pcloud: :class:`.PointCloud`
        :return: The predictions.
        :rtype: :class:`np.ndarray`
        """
        # Return the predictions
        return self.pps.predict(self.pipeline, pcloud)
