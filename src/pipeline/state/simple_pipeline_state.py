import copy

from src.pipeline.state.pipeline_state import PipelineState, \
    PipelineStateException
from src.pipeline.predictive_pipeline import PredictivePipeline
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.utils.ctransf.class_transformer import ClassTransformer
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
    :ivar base_pcloud: The pcloud given during initialization. While pcloud
        can be updated during iterations, base_pcloud is used as a baseline
        point cloud defining the initial value of any iteration.
    :vartype base_pcloud: :class:`.PointCloud`
    :ivar model: The model corresponding to the current pipeline state.
    :vartype model: :class:`.Model`
    :ivar base_model: The model given during initialization. While model can
        be updated during iterations, base_model is used as a baseline
        model defining the initial value of any iteration.
    :vartype base_model: :class:`.Model`
    :ivar fnames: The list of strings representing the feature names.
    :vartype fnames: list
    :ivar base_fnames: The fnames given during initialization. While fnames can
        be updated during iterations, base_fnames is used as a baseline
        list of feature names defining the initial value of any iteration.
    :vartype base_fnames: list
    :ivar preds: The predictions corresponding to the current pipeline state.
    :vartype preds: :class:`np.ndarray`
    :ivar base_preds: The preds given during initialization. While preds can
        be updated during iterations, base_preds is used as a baseline
        list of predictions defining the initial value of any iteration.
    :vartype base_preds: :class:`np.ndarray`
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
        self.base_pcloud = kwargs.get('pcloud', None)
        self.base_model = kwargs.get('model', None)
        self.base_fnames = kwargs.get('fnames', None)
        self.base_preds = kwargs.get('preds', None)
        self.pcloud, self.model, self.fnames, self.preds = [None]*4

    # ---  PIPELINE STATE METHODS  --- #
    # -------------------------------- #
    def _update(self, comp, **kwargs):
        """
        See :meth:`pipeline_state.PipelineState._update`.
        """
        # Extract key-word arguments of interest
        new_pcloud = kwargs.get('new_pcloud', None)
        new_model = kwargs.get('new_model', None)
        new_preds = kwargs.get('new_preds', None)
        # Handle the many component types
        if isinstance(comp, Miner):
            self.update_pcloud(comp, new_pcloud)  # Mine generated features
        elif isinstance(comp, Imputer):
            self.update_pcloud(comp, new_pcloud)  # Impute generated features
        elif isinstance(comp, FeatureTransformer):
            self.update_pcloud(comp, new_pcloud)  # Transform gen. feats.
        elif isinstance(comp, ClassTransformer):
            self.update_pcloud(comp, new_pcloud)  # Transform class (or pred.)
        elif isinstance(comp, ModelOp):
            self.update_model(comp, new_model)
            if comp.op == ModelOp.OP.TRAIN:
                pass
            elif comp.op == ModelOp.OP.PREDICT:
                self.update_preds(comp, new_preds)
            else:
                raise PipelineStateException(
                    'SimplePipelineState received an unexpected model '
                    f'operation: {comp.op}'
                )
        elif isinstance(comp, PredictivePipeline):
            self.update_preds(comp, new_preds)
            self.update_model(comp, new_model)
        else:
            raise PipelineStateException(
                'SimplePipelineState failed to update because an unexpected '
                f'component was received:\n{comp}'
            )
        # Return
        return self

    def prepare_iter(self, **kwargs):
        """
        See :meth:`pipeline_state.PipelineState.prepare_iter`.
        """
        self.fnames = kwargs.get('fnames', None)
        if self.fnames is None:
            self.fnames = copy.deepcopy(self.base_fnames)
        self.model = kwargs.get('model', None)
        if self.model is None:
            self.model = copy.deepcopy(self.base_model)
        self.pcloud = kwargs.get('pcloud', None)
        if self.pcloud is None:
            self.pcloud = copy.deepcopy(self.base_pcloud)
        self.preds = kwargs.get('preds', None)
        if self.preds is None:
            self.preds = copy.deepcopy(self.base_preds)

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
            LOGGING.LOGGER.debug(
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

    def update_preds(self, comp, new_preds):
        """
        Handle the update of the predictions.

        :param comp: The component that updated the predictions.
        :param new_preds: The new predictions.
        :return: Nothing but the pipeline state itself is updated.
        """
        # Check
        if new_preds is None:
            raise PipelineStateException(
                'SimplePipelineState cannot update predictions without '
                'new predictions. None were given.'
            )
        # Update predictions
        self.preds = new_preds
