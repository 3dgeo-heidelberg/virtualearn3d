# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pipeline import Pipeline, PipelineException


# ---   CLASS   --- #
# ----------------- #
class SequentialPipeline(Pipeline):
    """
    :author: Alberto M. Esmoris Pena
    Sequential pipeline (no loops, no recursion).
    See :class:`Pipeline`.
    """
    def __init__(self, **kwargs):
        """
        Initialize an instance of SequentialPipeline.
        A sequential pipeline execute the different components in the order
            they are given.
        :param **kwargs: The attributes for the SequentialPipeline
        """
        # Call parent init
        super().__init__(**kwargs)
        # Pipeline as sequential list of components
        # TODO Remove section ---
        print(f'SequentialPipeline in_pcloud: {self.in_pcloud}')
        print(f'SequentialPipeline out_pcloud: {self.out_pcloud}')
        # --- TODO Remove section
        # TODO Rethink : Implement

    # ---  RUN PIPELINE  --- #
    # ---------------------- #
    def run(self):
        """
        Run the sequential pipeline.
        :return: Nothing.
        """
        print(f'SequentialPipeline.run() TODO YET')  # TODO Remove
        # TODO Rethink : Implement