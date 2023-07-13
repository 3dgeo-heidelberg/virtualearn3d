# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.inout.model_writer import ModelWriter
from src.inout.predictive_pipeline_writer import PredictivePipelineWriter


# ---   CLASS   --- #
# ----------------- #
class WriterUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to handle writer objects.

    See :class:`.Writer`.
    """
    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_writer_class(spec):
        """
        Extract the writer's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a miner.
        :rtype: :class:`.Writer`
        """
        writer = spec.get('writer', None)
        if writer is None:
            raise ValueError(
                "Writing a point cloud requires a writer. None was specified."
            )
        # Check writer class
        writer_low = writer.lower()
        if writer_low == 'writer':
            return Writer
        if writer_low == 'modelwriter':
            return ModelWriter
        if writer_low == 'predictivepipelinewriter':
            return PredictivePipelineWriter
        # An unknown writer was specified
        raise ValueError(f'There is no known writer "{writer}"')
