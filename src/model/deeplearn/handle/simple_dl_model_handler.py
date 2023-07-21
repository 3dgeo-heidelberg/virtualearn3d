# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.report.deep_learning_model_summary_report import \
    DeepLearningModelSummaryReport
import src.main.main_logger as LOGGING
import tensorflow as tf
import copy
import time


# ---   CLASS   --- #
# ----------------- #
class SimpleDLModelHandler(DLModelHandler):
    """
    # TODO Rethink : Document class and ivars
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, arch, **kwargs):
        # TODO Rethink : doc
        # Call parent's init
        super().__init__(arch, **kwargs)
        # Assign member attributes
        self.summary_report_path = kwargs.get('summary_report_path', None)
        self.out_prefix = kwargs.get('out_prefix', None)
        self.training_epochs = kwargs.get('epochs', 100)
        self.history = kwargs.get('history', None)
        self.checkpoint_path = kwargs.get('checkpoint_path', None)
        self.checkpoint_monitor = kwargs.get('checkpoint_monitor', 'loss')
        self.compilation_args = kwargs.get('compilation_args', {
            'optimizer': tf.keras.optimizers.SGD(learning_rate=1e-3),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(),
            'metrics': [tf.keras.metrics.sparse_categorical_accuracy]
        })

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def _fit(self, X, y, F=None):
        # Report the model
        summary = DeepLearningModelSummaryReport(self.compiled)
        LOGGING.LOGGER.info(summary.to_string())
        if self.summary_report_path is not None:
            summary.to_file(
                self.summary_report_path,
                out_prefix=self.out_prefix
            )
            LOGGING.LOGGER.info(
                'Deep learning model summary written to "'
                f'{self.summary_report_path}"'
            )
        # Handle training callbacks
        callbacks = []
        if self.checkpoint_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path,
                monitor=self.checkpoint_monitor,
                save_best_only=True,
                save_weights_only=True
            ))
        # Fit the model
        start = time.perf_counter()
        self.history = self.compiled.fit(
            *self.arch.run_pre({'X': X, 'y': y}),
            epochs=self.training_epochs,
            callbacks=callbacks
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Deep learning model trained on {X.shape[0]} points during '
            f'{self.training_epochs} in {end-start:.3f} seconds.'
        )
        # Take best model from checkpoint
        if self.checkpoint_path is not None:
            self.compiled.load_weights(self.checkpoint_path)
        # Plot history
        # TODO Rethink : Implement
        # Return
        return self

    def _predict(self):
        # TODO Rethink : Implement
        pass

    def compile(self, X=None, y=None, F=None):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.compile`.
        """
        if not self.arch.is_built():
            self.arch.build()
        self.compiled = self.arch.nn
        self.compiled.compile(**self.compilation_args)
        return self
