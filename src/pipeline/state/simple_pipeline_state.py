from src.pipeline.state.pipeline_state import PipelineState, \
    PipelineStateException
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.model.model_op import ModelOp
import src.main.main_logger as LOGGING


# ---   CLASS   --- #
# ----------------- #
class SimplePipelineState(PipelineState):
    """
    :author: Alberto M. Esmoris Pena

    Simple pipeline state that accounts for current point cloud, feature names,
    and model.

    :ivar pcloud: The point cloud corresponding to the current pipeline state.
    :vartype pcloud: :class:`.PointCloud`
    :ivar model: The model corresponding to the current pipeline state.
    :vartype model: :class:`.Model`
    :ivar fnames: The list of strings representing the feature names.
    :vartype fnames: list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handle the initialization of a simple pipeline state.

        See :class`.PipelineState` and
        `meth`:`pipeline_state.PipelineState.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes to simple pipeline state
        self.pcloud = kwargs.get('pcloud', None)
        self.model = kwargs.get('model', None)
        self.fnames = kwargs.get('fnames', None)

    # ---  PIPELINE STATE METHODS  --- #
    # -------------------------------- #
    def _update(self, comp, **kwargs):
        """
        See :meth:`pipeline_state.PipelineState._update`.
        """
        # Extract key-word arguments of interest
        new_pcloud = kwargs.get('new_pcloud', None)
        new_model = kwargs.get('new_model', None)
        # Handle the many component types
        if isinstance(comp, Miner):
            self.update_pcloud(comp, new_pcloud)  # A miner generates features
        elif isinstance(comp, Imputer):
            self.update_pcloud(comp, new_pcloud)  # A miner generates features
        elif isinstance(comp, FeatureTransformer):
            self.update_pcloud(comp, new_pcloud)  # A miner generates features
        elif isinstance(comp, ModelOp):
            if comp.op == ModelOp.OP.TRAIN:
                self.update_model(comp, new_model)
        else:
            raise PipelineStateException(
                'SimplePipelineState failed to update because an unexpected '
                f'component was received:\n{comp}'
            )
        # Return
        return self

    # ---  SIMPLE PIPELINE STATE METHODS  --- #
    # --------------------------------------- #
    def update_pcloud(self, comp, new_pcloud):
        """
        Handle the update of the point cloud.

        :param comp: The component that updated the point cloud.
        :param new_pcloud: The new point cloud.
        :return: Nothing but the pipeline state itself is updated.
        """
        # Check
        if new_pcloud is None:
            raise PipelineStateException(
                'SimplePipelineState cannot update point cloud state without '
                'a point cloud. None was given.'
            )
        # Update point cloud
        self.pcloud = new_pcloud
        # Update fnames
        frenames = getattr(comp, 'frenames', None)
        if frenames is not None:
            self.fnames = frenames
        else:
            LOGGING.LOGGER.warning(
                'SimplePipelineState updated the point cloud without '
                'updating the feature names.'
            )

    def update_model(self, comp, new_model):
        """
        Handle the update of the model.

        :param comp: The component that updated the model.
        :param new_model: The new model.
        :return: Nothing but the pipeline state itself is updated.
        """
        # Check
        if new_model is None:
            raise PipelineStateException(
                'SimplePipelineState cannot update model state without '
                'a model. None was given.'
            )
        # Update model
        self.model = new_model
