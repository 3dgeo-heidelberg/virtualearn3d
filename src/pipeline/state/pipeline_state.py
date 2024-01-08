# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from abc import abstractmethod


# ---  EXCEPTIONS  --- #
# -------------------- #
class PipelineStateException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to the pipeline state.
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PipelineState:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface for any pipeline state and a common
        baseline implementation.

    :ivar step: The step of the pipeline. Typically, it is initialized to zero
        and updated for each call to the update method.
    :vartype step: int
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handle the root-level (most basic) initialization of any pipeline's
        state.

        :param kwargs: The attributes for the PipelineState.
        """
        self.step = 0

    # ---  PIPELINE STATE METHODS  --- #
    # -------------------------------- #
    def update(self, comp, **kwargs):
        """
        Update the pipeline's state for a given component
        (e.g, :class:`.Miner`, :class:`.ModelOp`, and :class:`.Imputer`) that
        has been executed in the pipeline.

        :param comp: The component that has been executed.
        :return: The updated pipeline state (also updated in place).
        """
        try:
            self._update(comp, **kwargs)
            self.step += 1  # Update step count
        except Exception as ex:
            raise PipelineStateException(
                'PipelineState could not be successfully updated.'
            ) from ex

    @abstractmethod
    def _update(self, comp, **kwargs):
        """
        The logic of the update.

        See :meth:`pipeline_state.PipelineState.update`.
        """
        pass

    def prepare_iter(self, **kwargs):
        """
        The logic to prepare an iteration. This method must handle any
        member attribute that needs to be updated at the beginning of each
        iteration.

        An iteration consists of applying the many steps of the pipeline to one
        case. For instance, the same sequential pipeline can be used to train
        models using two different datasets. In this case, the first model
        will be generated during the first iteration, while the second model
        will be generated during the second iteration.

        The prepare_iter method is necessary to have a pipeline state that does
        not propagate particularities of the first case to the second, e.g.,
        to avoid considering the last state of the feature names in the first
        iteration as the initial feature names in the second iteration.

        :param kwargs: The key-word arguments to prepare the iteration.
        :return: The pipeline state itself is updated and returned.
        :rtype: :class:`.PipelineState`
        """
        pass
