# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException


# ---  EXCEPTIONS  --- #
# -------------------- #
class PipelinePredictiveStrategyException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to predictive strategies for pipelines.
    See :class:`VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PipelinePredictiveStrategy:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface for any pipeline's predictive
    strategy.

    :ivar out_path: The output path for the predictive strategy. It can be
        None. If it is given, components that demand an output prefix will
        consider the output path as an output prefix.
    :vartype out_path: str
    :ivar external_state: The pipeline's state to be modified by the
        predictive strategy. If None, the updates will only happen inside the
        context of the pipeline predictive strategy. If not None, the updates
        will propagate to the given state. For instance, a predictive pipeline
        strategy that modifies a point cloud will override the point cloud
        in the state too.
    :vartype state: :class:`.PipelineState`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handles the root-level (most basic) initialization of any pipeline's
        predictive strategy.

        :param kwargs: The attributes for the pipeline's predictive strategy.
        """
        self.out_path = kwargs.get('out_path', None)
        self.external_state = kwargs.get('external_state', None)

    # ---  PIPELINE PREDICTIVE STRATEGY  --- #
    # -------------------------------------- #
    @abstractmethod
    def predict(self, pipeline, pcloud):
        """
        The predict method computes the predictions on the point cloud using
        the given pipeline.

        :param pipeline: The pipeline to compute the predictions.
        :type pipeline: :class:`.Pipeline`
        :param pcloud: The point cloud to be predicted.
        :type pcloud: :class:`.PointCloud`
        :return: The predictions.
        :rtype: :class:`np.ndarray`
        """
        pass
