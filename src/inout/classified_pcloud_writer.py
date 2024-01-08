# ---   IMPORTS   --- #
# ------------------- #
from src.inout.writer import Writer
from src.report.classified_pcloud_report import ClassifiedPcloudReport
import src.main.main_logger as LOGGING


# ---   CLASS   --- #
# ----------------- #
class ClassifiedPcloudWriter(Writer):
    """
    :author: Alberto M. Esmoris Pena

    Class for writing classified point clouds (mostly to be used in
    pipelines).

    See :class:`.Writer`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.Writer`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   WRITE   --- #
    # ----------------- #
    def write(self, state, prefix=None, info=True):
        """
        Write the classified point cloud.

        See :class:`.Writer` and :meth:`writer.Writer.write`.

        :param state: The pipeline's state that must contain the classified
            point cloud.
        :type state: :class:`.PipelineState`
        """
        # Prepare path
        path = self.prepare_path(prefix)
        # Find class names
        class_names = None
        if state.model is not None:
            model = getattr(state.model, "model", None)  # Model from ModelOp
            if model is not None:  # Take class names from model, if available
                class_names = getattr(model, "class_names", None)
        # Build report
        cp_report = ClassifiedPcloudReport(
            X=state.pcloud.get_coordinates_matrix(),
            y=state.pcloud.get_classes_vector(),
            yhat=state.preds,
            zhat=None,
            class_names=class_names
        )
        # Write report
        cp_report.to_file(path, prefix)
