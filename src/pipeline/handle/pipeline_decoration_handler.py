# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pipeline import PipelineException
import src.main.main_logger as LOGGING
from src.mining.fps_decorated_miner import FPSDecoratedMiner


# ---   CLASS   --- #
# ⨪---------------- #
class PipelineDecorationHandler:
    """
    :author: Alberto M. Esmoris Pena

    A handler for the extra logic needed by decorated components to work
    correctly.

    See :class:`.FPSDecoratorTransformer`, :class:`.FPSDecoratedModel`, and
    :class:`.FPSDecoratedMiner`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handles the initialization of a pipeline decoration handler for
        sequential pipelines .

        :param kwargs: The attributes for the pipeline's decoration handler.
        """
        self.out_prefix = kwargs.get('out_prefix', None)

    # ---   HANDLE   --- #
    # ------------------ #
    def handle_preprocess_decoration(self, state, comp, comp_id, comps):
        """
        Handle the decoration of the pipeline at the preprocess stage.

        See :class:`.PipelineExecutor`, :meth:`.PipelineExecutor.__call__`, and
        :meth:`.PipelineExecutor.pre_process`.

        :return: Nothing, modifications (if needed) will happen in the received
            input objects directly.
        """
        # Handle decorated miners
        if isinstance(comp, FPSDecoratedMiner):
            comp.out_prefix = self.out_prefix  # Propagate output prefix
